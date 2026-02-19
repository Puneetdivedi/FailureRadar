"""
app.py — IndustrialCopilot: Streamlit RAG chatbot for industrial sensor diagnostics.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.retriever import Retriever
from src.llm import ask_ollama, build_rag_prompt

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndustrialCopilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin: 0.5rem 0 0; }

    .metric-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: bold; color: #e94560; }
    .metric-card .label { font-size: 0.85rem; color: #a8b2d8; }

    .status-critical { color: #ff4757; font-weight: bold; }
    .status-warning  { color: #ffa502; font-weight: bold; }
    .status-normal   { color: #2ed573; font-weight: bold; }

    .chat-user     { background: #0f3460; border-radius: 10px; padding: 0.8rem; margin: 0.5rem 0; }
    .chat-bot      { background: #16213e; border: 1px solid #e94560; border-radius: 10px; padding: 0.8rem; margin: 0.5rem 0; }
    .context-box   { background: #0d1117; border-left: 3px solid #e94560; padding: 0.8rem; border-radius: 6px; font-size: 0.8rem; color: #8b949e; }

    .sidebar .sidebar-content { background: #1a1a2e; }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white; border: none; border-radius: 8px; width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Retriever (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_retriever():
    try:
        return Retriever(), None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_sensor_data():
    try:
        return pd.read_csv("data/sensor_logs.csv")
    except:
        return pd.DataFrame()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏭 IndustrialCopilot</h1>
    <p>RAG-Powered Sensor Diagnostics Assistant · Ollama · FAISS · Local LLM</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    ollama_model = st.selectbox(
        "Ollama Model",
        ["llama3.2", "mistral", "gemma2", "llama3.1", "phi3"],
        help="Must be pulled via: ollama pull <model>"
    )

    top_k = st.slider("Retrieved Chunks (top_k)", 3, 10, 6,
                       help="More chunks = richer context but slower")

    show_context = st.checkbox("Show Retrieved Context", value=False,
                                help="Debug: see what RAG retrieved")

    st.markdown("---")
    st.markdown("### 📋 Quick Queries")
    quick_queries = [
        "Why did MOTOR-01 overheat?",
        "Which machine had the most critical events?",
        "What caused PUMP-05 bearing fault?",
        "Summarize all anomalies for COMPRESSOR-03",
        "Which equipment needs immediate maintenance?",
        "Compare vibration levels across all machines",
    ]
    for q in quick_queries:
        if st.button(q, key=f"quick_{q[:20]}"):
            st.session_state["prefill_query"] = q

    st.markdown("---")
    st.markdown("### 🔧 Setup")
    st.code("ollama serve\nollama pull llama3.2\npython src/ingest.py\nstreamlit run app.py", language="bash")

# ── Load data & retriever ────────────────────────────────────────────────────
df = load_sensor_data()
retriever, retriever_err = load_retriever()

# ── Sensor Overview Dashboard ────────────────────────────────────────────────
if not df.empty:
    st.markdown("### 📊 Live Sensor Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    total     = len(df)
    critical  = len(df[df["status"] == "critical"])
    warning   = len(df[df["status"] == "warning"])
    normal    = len(df[df["status"] == "normal"])
    machines  = df["machine_id"].nunique()

    with col1:
        st.markdown(f'<div class="metric-card"><div class="value">{machines}</div><div class="label">Machines</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="value">{total}</div><div class="label">Total Readings</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="value" style="color:#ff4757">{critical}</div><div class="label">Critical Events</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="value" style="color:#ffa502">{warning}</div><div class="label">Warnings</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="value" style="color:#2ed573">{normal}</div><div class="label">Normal</div></div>', unsafe_allow_html=True)

    st.markdown("&nbsp;")

    # Machine status table
    tab1, tab2 = st.tabs(["📈 Recent Sensor Readings", "🚨 Alerts Only"])
    with tab1:
        display_df = df.copy()
        def color_status(val):
            if val == "critical": return "background-color: #3d0000; color: #ff4757"
            if val == "warning":  return "background-color: #3d2000; color: #ffa502"
            return "background-color: #003d00; color: #2ed573"
        st.dataframe(
            display_df.tail(20).style.applymap(color_status, subset=["status"]),
            use_container_width=True, height=280
        )
    with tab2:
        alerts_df = df[df["alert"] != "none"]
        st.dataframe(alerts_df.style.applymap(color_status, subset=["status"]),
                     use_container_width=True, height=280)

st.markdown("---")

# ── Chat Interface ────────────────────────────────────────────────────────────
st.markdown("### 🤖 Ask IndustrialCopilot")

if retriever_err:
    st.warning(f"⚠️ Vector store not found. Run `python src/ingest.py` first.\n\nError: {retriever_err}")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">👷 <b>Engineer:</b> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">🤖 <b>IndustrialCopilot:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        if show_context and "context" in msg:
            with st.expander("📎 Retrieved Context"):
                st.markdown(f'<div class="context-box">{msg["context"]}</div>', unsafe_allow_html=True)

# Input
prefill = st.session_state.pop("prefill_query", "")
user_input = st.text_input(
    "Ask about your plant equipment...",
    value=prefill,
    placeholder="e.g. Why did MOTOR-01 overheat? Which machine needs urgent maintenance?",
    key="user_input"
)

col_send, col_clear = st.columns([5, 1])
with col_send:
    send = st.button("🔍 Analyze", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear"):
        st.session_state.messages = []
        st.rerun()

if send and user_input.strip() and retriever:
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔎 Retrieving sensor data... 🧠 Analyzing with LLM..."):
        # RAG retrieve
        docs    = retriever.retrieve(question, top_k=top_k)
        context = retriever.format_context(docs)
        prompt  = build_rag_prompt(question, context)

        # LLM call
        answer  = ask_ollama(prompt, model=ollama_model)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context.replace("\n", "<br>")
    })
    st.rerun()

elif send and not retriever:
    st.error("Please run `python src/ingest.py` to build the vector store first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#555; font-size:0.8rem;'>"
    "IndustrialCopilot · RAG + Ollama + FAISS · Built by Puneet Divedi"
    "</center>",
    unsafe_allow_html=True
)
