# 🤖 Agentic RAG Chatbot with Long-Term Memory

An intelligent, tool-using chatbot built with **LangGraph** that combines Retrieval-Augmented Generation (RAG) over user-uploaded PDFs, persistent long-term memory across sessions, and access to external tools — all wrapped in a clean **Streamlit** interface.

Unlike a simple chatbot, this project is an **agent**: it decides when to search the web, query a PDF, fetch a stock price, do a calculation, or call an MCP server tool — and it remembers relevant facts about the user across conversations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Long-Term Memory** | Learns and stores facts about the user across sessions using a persistent SQLite-backed memory store |
| 💬 **Short-Term Memory** | Keeps track of the ongoing conversation per chat thread |
| 📄 **PDF-based RAG** | Upload a PDF and ask questions about its content — powered by FAISS vector search |
| 🌐 **Web Search** | Fetches real-time information from the internet using Tavily |
| 📈 **Stock Price Lookup** | Gets live stock quotes using the Alpha Vantage API |
| 🧮 **Calculator** | Performs basic arithmetic operations |
| 💰 **Expense Tracker (MCP)** | A separate MCP (Model Context Protocol) server that lets the bot add, update, delete, and summarize your expenses |
| 🔀 **Multi-threaded Conversations** | Create and switch between multiple independent chat sessions |
| ⚡ **Streaming Responses** | Answers stream in token-by-token for a smooth, real-time chat feel |

---

## 🏗️ Architecture

The core of the system is a LangGraph state graph:

```
START
  │
  ▼
remember_node   ──►  extracts new long-term memories from the latest user message
  │
  ▼
chat_node       ──►  generates a response, personalized using stored memory,
  │                  and decides whether to call a tool
  │
  ├──► tools_condition ──► tools (web_search / get_stock_price / calculator /
  │         │                     rag_tool / MCP tools)
  │         │
  │         └──────────────► back to chat_node
  │
  ▼
END
```

**Key design choices:**
- **Memory and chat are decoupled** into separate nodes so memory extraction (via a structured-output LLM call) doesn't pollute the main conversation context.
- **Per-thread RAG indexes** mean each chat conversation can reference its own uploaded document without interference from other threads.
- **Checkpointing with `AsyncSqliteSaver`** gives durable conversation history and the ability to list/resume past threads.
- **`InMemoryStore`** holds long-term memory namespaced by user ID, separate from short-term conversation state.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://www.langchain.com/langgraph) (`StateGraph`, conditional tool routing) |
| LLM | OpenAI `gpt-4.1-mini` (chat), `gpt-4o-mini` (memory extraction) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Web search | Tavily |
| Stock data | Alpha Vantage |
| External tools | [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) via `langchain-mcp-adapters` |
| Short-Term Memory | `SqliteSaver` |
| Long-Term Memory | `SqliteStore` |
| Frontend | [Streamlit](https://streamlit.io/) |
| Vector store | FAISS |

---

## 📁 Project Structure

```
Agentic-RAG-Chatbot-with-Long-Term-Memory/
│
├── langgraph_tool_backend.py      # Core agent logic: graph, tools, memory, RAG
├── streamlit_frontend_tool.py     # Streamlit chat UI, PDF upload, threading
├── requirements.txt               # Python dependencies
├── .env.example                   # Example environment variables file
├── .gitignore
├── LICENSE
│
└── expense-tracker-mcp-server/    # Standalone MCP server for expense tracking
    ├── main.py                    # MCP tool definitions (add/list/update/delete/summarize)
    ├── categories.json            # Allowed expense categories & subcategories
    ├── pyproject.toml
    ├── uv.lock
    └── .python-version
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- API keys for:
  - [OpenAI](https://platform.openai.com/) (for the LLM and embeddings)
  - [Tavily](https://tavily.com/) (for web search)
  - [Alpha Vantage](https://www.alphavantage.co/) (for stock prices)

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/agentic-rag-chatbot.git
cd agentic-rag-chatbot
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Copy the example file and fill in your keys:
```bash
cp .env.example .env
```

```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_key_here

```

### 4. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## 💡 Usage

1. **Start a new chat** from the sidebar, or resume a past conversation.
2. **Upload a PDF** (optional) to enable document-grounded Q&A for that thread.
3. **Chat naturally** — ask general questions, request stock prices, do math, or ask about the uploaded document. The agent decides which tool(s) to use.
4. Over time, the assistant **remembers stable facts about you** (name, projects, preferences) and personalizes its responses and greetings accordingly.

---
