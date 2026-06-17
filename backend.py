from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from os import getenv
import aiosqlite
import requests
import asyncio
import threading

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
_llm_configured = bool(getenv("LLM_API_KEY") and getenv("LLM_API_URL"))

llm = ChatOpenAI(
    base_url=getenv("LLM_API_URL", "http://placeholder"),
    api_key=getenv("LLM_API_KEY", "placeholder"),
    model="auto"
) if _llm_configured else None

# -------------------
# 2. Tools
# -------------------
search_tool = TavilySearchResults(max_results=10) if getenv("TAVILY_API_KEY") else None


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={getenv('STOCK_API')}"
    r = requests.get(url)
    return r.json()

filesystem_dirs = [d.strip() for d in getenv("FILESYSTEM_DIRS", "").split(",") if d.strip()]

client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        },
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                *filesystem_dirs,
            ],
            "transport": "stdio",
        },
        "github": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-github"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
            }
        },
        "gmail": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@sowonai/mcp-gmail"],
            "env": {
                "GOOGLE_CLIENT_ID": getenv("CLIENT_ID"),
                "GOOGLE_CLIENT_SECRET": getenv("CLIENT_SECRET"),
                "GOOGLE_REFRESH_TOKEN": getenv("REFRESH_TOKEN"),
            }
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools() if _llm_configured else []

tools = [t for t in [search_tool, get_stock_price, *mcp_tools] if t is not None] if _llm_configured else []
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    if not llm_with_tools:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="⚠️ LLM not configured. Please set your API keys in Settings and restart.")]}
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    seen = set()
    threads = []

    async for checkpoint in checkpointer.alist(None):
        tid = checkpoint.config["configurable"]["thread_id"]

        if tid not in seen:
            seen.add(tid)
            threads.append(tid)

    return threads


def retrieve_all_threads():
    return run_async(_alist_threads())