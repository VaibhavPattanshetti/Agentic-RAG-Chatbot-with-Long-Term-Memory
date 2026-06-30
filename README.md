# 🤖 Agentic RAG Chatbot with Long-Term Memory

An intelligent, tool-using chatbot built with **LangGraph** that combines Retrieval-Augmented Generation (RAG) over user-uploaded PDFs, persistent long-term memory across sessions, and access to external tools — all wrapped in a clean **Streamlit** interface.

Unlike a simple chatbot, this project is an **agent**: it decides when to search the web, query a PDF, fetch a stock price, do a calculation, or call an MCP server tool — and it remembers relevant facts about the user across conversations.

---

## ✨ Features

- **🧠 Long-Term Memory** — Automatically extracts and stores stable facts about the user (name, preferences, ongoing projects) in a memory store, then injects them into the system prompt to personalize every response. Duplicate/redundant memories are filtered out using an LLM-based memory extractor.
- **📄 Agentic RAG over PDFs** — Upload a PDF per chat thread; it's chunked, embedded, and indexed with FAISS. The agent decides on its own when to call the `rag_tool` to retrieve relevant context instead of always forcing retrieval.
- **🔧 Tool-Calling Agent** — Built on LangGraph's `ToolNode` and `tools_condition`, the agent can autonomously choose between:
  - `web_search` (via Tavily) for current/general information
  - `get_stock_price` (via Alpha Vantage) for live stock quotes
  - `calculator` for arithmetic
  - `rag_tool` for document-grounded answers
  - **MCP server tools** (e.g., a custom expense tracker) via `langchain-mcp-adapters`
- **💬 Multi-Thread Conversations** — Each chat has its own thread ID, persisted via `AsyncSqliteSaver` checkpointing, so conversations can be resumed, switched, and listed from the sidebar.
- **⚡ Streaming Responses** — Assistant tokens stream live in the UI, with tool usage shown via real-time status indicators (e.g., "🔧 Using `web_search` …").
- **🔌 MCP Integration** — Demonstrates connecting to a custom [Model Context Protocol](https://modelcontextprotocol.io/) server to extend the agent with external, stateful tools (e.g., an expense tracker).

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
| Agent orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM | OpenAI `gpt-4.1-mini` (chat), `gpt-4o-mini` (memory extraction) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | FAISS |
| Web search | Tavily |
| Stock data | Alpha Vantage |
| External tools | MCP (Model Context Protocol) via `langchain-mcp-adapters` |
| Persistence | SQLite (`aiosqlite` + LangGraph `AsyncSqliteSaver`) |
| Frontend | Streamlit |

---

## 📁 Project Structure

```
agentic-rag-chatbot/
├── app.py                  # Streamlit frontend (chat UI, PDF upload, thread management)
├── backend/
│   └── graph.py             # LangGraph backend: nodes, tools, memory, checkpointing
├── faiss_indexes/           # Per-thread FAISS vector stores (generated at runtime, gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- API keys for OpenAI and Tavily (and optionally Alpha Vantage)

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

# Optional — only needed if running the expense tracker MCP server
EXPENSE_TRACKER_PYTHON=
EXPENSE_TRACKER_SCRIPT=
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

## 🗺️ Roadmap / Ideas for Extension

- [ ] Swap `InMemoryStore` for a persistent long-term memory backend (e.g., Postgres or Redis)
- [ ] Add memory editing/deletion via the UI
- [ ] Support multiple documents per thread with source attribution in answers
- [ ] Add evaluation/observability (e.g., LangSmith tracing)
- [ ] Containerize with Docker for easier deployment

---
