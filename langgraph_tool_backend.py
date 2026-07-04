from __future__ import annotations

import asyncio
import os
import requests
import sqlite3
import tempfile
import threading
import sys
from datetime import date

from typing import Annotated, Any, Dict, Optional, TypedDict

import aiosqlite
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool, BaseTool

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig

import uuid

from typing import List
from pydantic import BaseModel, Field

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore



############ LLM ############
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



############ PDF retriever store (per thread) ############


# folder for storing vector stores
FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)


_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)

        index_path = os.path.join(FAISS_DIR, str(thread_id))
        vector_store.save_local(index_path)

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _get_retriever(thread_id):

    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    index_path = os.path.join(FAISS_DIR, str(thread_id))

    if os.path.exists(index_path):

        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        _THREAD_RETRIEVERS[thread_id] = retriever

        return retriever

    return None



############ Tools ############

from langchain_tavily import TavilySearch

tavily = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

@tool
def web_search(query: str) -> str:

    """
    Search the web for recent and general information using Tavily.
    Returns the most relevant search results.
    """

    result = tavily.invoke(query)

    if not result.get("results"):
        return "No relevant web search results found."

    formatted = []

    for r in result["results"]:

        formatted.append(
            f"""
        Source: {r.get('url')}

        Title: {r.get('title')}

        Summary:
        {r.get('content')}
        """
        )

    return "\n\n".join(formatted)

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key from environment.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY not set in environment."}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}



@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF.
    """

    thread_id = config["configurable"]["thread_id"]

    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat.",
            "query": query,
        }

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {                        # this return is send to model
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
    }


MCP_SERVER_PATH = os.path.join(
    os.path.dirname(__file__), "expense-tracker-mcp-server", "main.py"
)  # adjust to wherever main.py actually lives relative to this file

mcp_client = MultiServerMCPClient({
    "expense_tracker": {
        "command": sys.executable,
        "args": [MCP_SERVER_PATH],
        "transport": "stdio",
    }
})

from langchain_core.tools import StructuredTool

def make_sync(async_tool):
    def sync_func(**kwargs):
        return asyncio.run(async_tool.coroutine(**kwargs))
    return StructuredTool(
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
        func=sync_func,
    )

mcp_tools_raw = asyncio.run(mcp_client.get_tools())
mcp_tools = [make_sync(t) for t in mcp_tools_raw]

tools = [web_search, get_stock_price, calculator, rag_tool] + mcp_tools
llm_with_tools = llm.bind_tools(tools)




############ State ############
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]




# ----------------------------
# Long Term Memory
# ----------------------------

from langgraph.store.sqlite import SqliteStore


store_conn = sqlite3.connect(
    "ltm.db",
    check_same_thread=False,
    isolation_level=None,   # autocommit mode — required so SqliteStore's own BEGIN/COMMIT works
)
store = SqliteStore(store_conn)
store.setup()

def print_store(user_id: str):
    ns = ("user", user_id, "details")

    items = store.search(ns)

    print("\n===== MEMORY =====")
    for item in items:
        print(item.key, "->", item.value)
    print("==================\n")


# ----------------------------
# System prompt
# ----------------------------
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored
assistance that reflects the user’s preferences, context, and past interactions.

If the user’s name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Vaibhav...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible.

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

Today's date is {current_date}. Resolve "today", "yesterday", "this week" etc. relative to this date before calling expense tools.

The user’s memory (which may be empty) is provided as: {user_details_content}
"""

# ----------------------------
# Memory extraction LLM
# ----------------------------
memory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

memory_extractor = memory_llm.with_structured_output(MemoryDecision)



MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""

# ----------------------------
############ Nodes ############
# ----------------------------
def remember_node(state: ChatState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # existing memory
    items = store.search(ns)
    existing = "\n".join(it.value["data"] for it in items) if items else "(empty)"

    # last user message
    last_msg = next(
        msg for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_msg.content},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(ns, str(uuid.uuid4()), {"data": mem.text})

    print_store(user_id)

    return {}  # no message change


def chat_node(state: ChatState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    user_details = "\n".join(it.value["data"] for it in items) if items else ""

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(
            user_details_content=user_details or "(empty)",
            current_date=date.today().isoformat()
        )
    )

    response =  llm_with_tools.invoke([system_msg] + state["messages"])
    
    return {"messages": [response]}




#tool node
tool_node = ToolNode(tools)


#checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn)    # Checkpointer


#graph
graph = StateGraph(ChatState)
graph.add_node("remember_node", remember_node)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "remember_node")
graph.add_edge("remember_node", "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer, store=store)




# helper function
def retrive_all_threads():

    all_threads = set() # want only unique thread ids, so using a set

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

