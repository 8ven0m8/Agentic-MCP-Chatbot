import queue
import uuid
from pathlib import Path

import streamlit as st
from backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv, set_key, dotenv_values

# ======================= Page Config ==========================
st.set_page_config(
    page_title="MCP Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ENV_FILE = Path(".env")

# ======================= Inject CSS ==========================
def load_css(path: str) -> None:
    css = Path(path).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("style.css")

# Header bar — logo + title centered
st.markdown("""
<div id="custom-topbar" style="
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    padding: 0.5rem 1.5rem;
    display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
    font-family: 'Inter', sans-serif;
    height: 44px; box-sizing: border-box;
">
    <span></span>
    <div style="display:flex; align-items:center; gap:0.75rem; justify-content:center;">
        <span class="topbar-mark" style="font-size:1.1rem;">⬡</span>
        <span class="topbar-brand" style="font-size:0.82rem; font-weight:600; letter-spacing:0.02em;">MCP Agent</span>
        <span class="topbar-badge" style="
            font-size:0.6rem; font-weight:500; letter-spacing:0.1em;
            text-transform:uppercase;
            border-radius:4px; padding:0.1rem 0.4rem;
        ">Re-Act · LangGraph</span>
    </div>
    <span class="topbar-version" style="font-size:0.65rem; font-family:'JetBrains Mono',monospace; text-align:right;">v1.0</span>
</div>
<div style="height:2.75rem;"></div>
""", unsafe_allow_html=True)

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_thread_state_and_timestamp(thread_id):
    """Fetches a thread's state once and returns (state, timestamp), so the
    sidebar can sort by recency without re-querying per thread."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state, (state.created_at or "")
    except Exception:
        return None, ""


# ======================= Settings Dialog ==========================
ENV_FIELDS = [
    # (key, label, type, placeholder, section)
    ("LLM_API_KEY",                 "LLM_API_KEY",                 "password", "sk-proj-4M3bRtY7xZ...",              "LLM Provider"),
    ("LLM_API_URL",                 "LLM_API_URL",                 "default",  "https://…",                          "LLM Provider"),
    ("LANGSMITH_API_KEY",           "LANGSMITH_API_KEY",           "password", "ls__…",                              "LangSmith / Observability"),
    ("LANGCHAIN_TRACING_V2",        "LANGCHAIN_TRACING_V2",        "default",  "true",                               "LangSmith / Observability"),
    ("LANGCHAIN_ENDPOINT",          "LANGCHAIN_ENDPOINT",          "default",  "https://api.smith.langchain.com",     "LangSmith / Observability"),
    ("LANGCHAIN_PROJECT",           "LANGCHAIN_PROJECT",           "default",  "my-project",                         "LangSmith / Observability"),
    ("STOCK_API",                   "STOCK_API",                   "password", "Take it from Alphavantage.co",       "Tool APIs"),
    ("TAVILY_API_KEY",              "TAVILY_API_KEY",              "password", "tvly-…",                             "Tool APIs"),
    ("GITHUB_PERSONAL_ACCESS_TOKEN","GITHUB_PERSONAL_ACCESS_TOKEN","password", "ghp_…",                              "Tool APIs"),
    ("CLIENT_ID",                   "CLIENT_ID",                   "default",  "*.apps.googleusercontent.com",       "Gmail OAuth"),
    ("CLIENT_SECRET",               "CLIENT_SECRET",               "password", "GOCSPX-…",                          "Gmail OAuth"),
    ("REFRESH_TOKEN",               "REFRESH_TOKEN",               "password", "1//…",                               "Gmail OAuth"),
    ("FILESYSTEM_DIRS",             "FILESYSTEM_DIRS",             "default",  "/path/one,/path/two",                "Filesystem"),
]

SECTIONS = ["LLM Provider", "LangSmith / Observability", "Tool APIs", "Gmail OAuth", "Filesystem"]

SECTION_ICONS = {
    "LLM Provider":               "🤖",
    "LangSmith / Observability":  "🔭",
    "Tool APIs":                  "🔧",
    "Gmail OAuth":                "📧",
    "Filesystem":                 "📁",
}

def section_divider(label: str):
    icon = SECTION_ICONS.get(label, "")
    st.markdown(f"""
    <div class="settings-section">
        <span class="icon">{icon}</span>
        <span class="label">{label}</span>
        <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)


@st.dialog("⚙  Environment Config", width="large")
def settings_dialog():
    current = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}

    st.markdown("""
    <div class="settings-hint">
        Changes are written to <span class="hl">.env</span>
        in your project root. Restart the app to apply.
    </div>
    """, unsafe_allow_html=True)

    values: dict[str, str] = {}
    current_section = None

    for key, label, field_type, placeholder, section in ENV_FIELDS:
        if key == "FILESYSTEM_DIRS":
            continue  # rendered separately below
        if section != current_section:
            section_divider(section)
            current_section = section

        col_label, col_input = st.columns([1, 2])
        with col_label:
            st.markdown(f'<div class="settings-field-label">{label}</div>', unsafe_allow_html=True)
        with col_input:
            val = st.text_input(
                label=label,
                value=current.get(key, ""),
                placeholder=placeholder,
                type=field_type,
                key=f"env_{key}",
                label_visibility="collapsed",
            )
        values[key] = val

    # ── Filesystem dirs section ──
    section_divider("Filesystem")

    raw_dirs = current.get("FILESYSTEM_DIRS", "")
    existing = [d.strip() for d in raw_dirs.split(",") if d.strip()]

    if "fs_dirs" not in st.session_state:
        st.session_state["fs_dirs"] = existing if existing else [""]

    st.markdown('<div class="settings-field-label">Allowed Directories</div>', unsafe_allow_html=True)

    for i, d in enumerate(st.session_state["fs_dirs"]):
        col_input, col_remove = st.columns([5, 1])
        with col_input:
            st.session_state["fs_dirs"][i] = st.text_input(
                label=f"dir_{i}",
                value=d,
                placeholder="/home/user/Documents",
                key=f"fs_dir_{i}",
                label_visibility="collapsed",
            )
        with col_remove:
            if st.button("✕", key=f"rm_dir_{i}", use_container_width=True):
                st.session_state["fs_dirs"].pop(i)
                st.rerun()

    if st.button("＋  Add directory", use_container_width=False):
        st.session_state["fs_dirs"].append("")
        st.rerun()

    # ── Save / Clear / Cancel ──
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    col_save, col_clear, col_cancel = st.columns([2, 1, 1])

    with col_save:
        if st.button("💾  Save to .env", use_container_width=True, type="primary"):
            ENV_FILE.touch(exist_ok=True)
            saved = 0
            for key, val in values.items():
                if val.strip():
                    set_key(str(ENV_FILE), key, val.strip())
                    saved += 1
            # Save filesystem dirs
            dirs_val = ",".join(d.strip() for d in st.session_state["fs_dirs"] if d.strip())
            if dirs_val:
                set_key(str(ENV_FILE), "FILESYSTEM_DIRS", dirs_val)
                saved += 1
            load_dotenv(ENV_FILE, override=True)
            st.success(f"✓ Saved {saved} variable(s). Restart app to apply.", icon="✅")

    with col_clear:
        if st.button("🗑  Clear", use_container_width=True):
            if ENV_FILE.exists():
                for key, _, _, _, _ in ENV_FIELDS:
                    if key != "LLM_API_KEY":
                        set_key(str(ENV_FILE), key, "")
            st.session_state["fs_dirs"] = [""]
            st.rerun()

    with col_cancel:
        if st.button("✕  Close", use_container_width=True):
            st.rerun()


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1rem; margin-top:0.5rem;">
        <div style="font-family:'Inter',sans-serif; font-size:0.65rem; font-weight:600;
                    letter-spacing:0.12em; text-transform:uppercase; color:var(--text-muted);
                    margin-bottom:0.75rem;">
            Workspace
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋  New conversation", use_container_width=True):
        reset_chat()

    st.markdown("""
    <div style="font-family:'Inter',sans-serif; font-size:0.65rem; font-weight:500;
                letter-spacing:0.1em; text-transform:uppercase; color:var(--text-muted);
                margin-top:1.5rem; margin-bottom:0.5rem; padding-left:0.1rem;">
        Recent
    </div>
    """, unsafe_allow_html=True)

    # Fetch each thread's state once, then sort by actual last-activity
    # timestamp so the newest conversation always appears at the top —
    # regardless of how chat_threads happens to be ordered.
    thread_data = [
        (thread_id, *get_thread_state_and_timestamp(thread_id))
        for thread_id in st.session_state["chat_threads"]
    ]
    thread_data.sort(key=lambda item: item[2], reverse=True)

    if not thread_data:
        st.markdown("""
        <div style="font-size:0.72rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;
                    padding:0.5rem 0.25rem;">
            No conversations yet.
        </div>
        """, unsafe_allow_html=True)
    else:
        for thread_id, state, _ in thread_data:
            convo = state.values.get("messages", []) if state else []
            if convo:
                raw = convo[0].content
                if isinstance(raw, list):
                    raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
                label = f"{raw[:32]}…" if len(raw) > 32 else raw
            else:
                label = "New chat"

            active = st.session_state["thread_id"] == thread_id
            if st.button(
                f"{'▸ ' if active else '  '}{label}",
                key=str(thread_id),
                use_container_width=True,
            ):
                st.session_state["thread_id"] = thread_id
                messages = load_conversation(thread_id)
                temp_messages = []
                for message in messages:
                    if isinstance(message, HumanMessage):
                        role, content = "user", message.content
                    elif isinstance(message, AIMessage):
                        if isinstance(message.content, str):
                            role, content = "assistant", message.content
                        elif isinstance(message.content, list):
                            role = "assistant"
                            content = "".join(
                                block.get("text", "")
                                for block in message.content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                        else:
                            continue
                    else:
                        continue
                    if content:
                        temp_messages.append({"role": role, "content": content})
                st.session_state["message_history"] = temp_messages

    # ── Connected tools badge strip ──
    st.markdown("""
    <div style="border-top:1px solid var(--border-soft); padding-top:0.85rem; margin-top:1rem;">
        <div class="tools-strip-label" style="font-size:0.65rem; font-family:'JetBrains Mono',monospace;
                    margin-bottom:0.5rem;">Connected tools</div>
        <div style="display:flex; flex-wrap:wrap; gap:0.3rem;">
            <span class="tool-badge" style="font-size:0.6rem; border-radius:4px; padding:0.15rem 0.4rem;
                         font-family:'JetBrains Mono',monospace;">tavily</span>
            <span class="tool-badge" style="font-size:0.6rem; border-radius:4px; padding:0.15rem 0.4rem;
                         font-family:'JetBrains Mono',monospace;">filesystem</span>
            <span class="tool-badge" style="font-size:0.6rem; border-radius:4px; padding:0.15rem 0.4rem;
                         font-family:'JetBrains Mono',monospace;">github</span>
            <span class="tool-badge" style="font-size:0.6rem; border-radius:4px; padding:0.15rem 0.4rem;
                         font-family:'JetBrains Mono',monospace;">gmail</span>
            <span class="tool-badge" style="font-size:0.6rem; border-radius:4px; padding:0.15rem 0.4rem;
                         font-family:'JetBrains Mono',monospace;">stocks</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings button pinned at bottom ──
    st.markdown("<div style='flex:1; min-height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid var(--border-soft); padding-top:0.75rem; margin-top:0.5rem;"></div>
    """, unsafe_allow_html=True)

    if st.button("⚙  Settings", use_container_width=True, key="open_settings"):
        settings_dialog()

# ============================ Main UI ============================
if not st.session_state["message_history"]:
    st.markdown("""
    <div class="empty-state">
        <div class="logo-mark">⬡</div>
        <div style="color:var(--text-primary); font-size:0.95rem; font-weight:500; margin-top:0.25rem;">
            What can I help you with?
        </div>
        <div style="color:var(--text-muted); font-size:0.75rem; max-width:320px; line-height:1.6;">
            I can browse the web, read files, check GitHub, send emails,
            look up stock prices, and more — just ask.
        </div>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask anything…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"⚙ `{tool_name}`", expanded=True
                        )
                        with status_holder["box"]:
                            st.markdown(f"`{tool_name}`")
                    else:
                        status_holder["box"].update(
                            label=f"⚙ `{tool_name}`",
                            state="running",
                            expanded=True,
                        )
                        with status_holder["box"]:
                            st.markdown(f"`{tool_name}`")

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    if isinstance(message_chunk.content, str):
                        yield message_chunk.content
                    elif isinstance(message_chunk.content, list):
                        for block in message_chunk.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                yield block["text"]

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✓ Done", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )