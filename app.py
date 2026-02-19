"""
app.py — FailureRadar v2
  ✅ Premium dark UI
  ✅ Multi-file CSV upload
  ✅ Isolation Forest anomaly detection
  ✅ PDF / DOCX report export
  ✅ RAG + Ollama chat diagnostics

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys, os, datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.retriever import Retriever
from src.llm       import ask_ollama, build_rag_prompt
from src.anomaly   import run_anomaly_detection, get_anomaly_summary
from src.report    import generate_pdf_report, generate_docx_report

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FailureRadar", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.stApp{background-color:#f5f7fa;color:#1a1a2e}
[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid #e2e8f0}
.fr-header{background:linear-gradient(135deg,#ffffff,#f0f4ff,#e8eeff);border:1px solid #e2e8f0;
  border-bottom:3px solid #e94560;padding:1.8rem 2rem;border-radius:12px;
  margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem;
  box-shadow:0 2px 12px rgba(0,0,0,0.06)}
.fr-header .logo{font-size:3rem}
.fr-header h1{color:#e94560;font-size:2rem;margin:0;letter-spacing:1px}
.fr-header p{color:#64748b;font-size:0.85rem;margin:0.2rem 0 0}
.fr-badge{display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;
  border-radius:20px;padding:2px 10px;font-size:0.72rem;color:#3b82f6;margin-right:4px}
.metric-row{display:flex;gap:1rem;margin-bottom:1.2rem}
.mc{flex:1;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
  padding:1rem 0.8rem;text-align:center;transition:all 0.2s;
  box-shadow:0 1px 4px rgba(0,0,0,0.05)}
.mc:hover{border-color:#e94560;box-shadow:0 4px 12px rgba(233,69,96,0.1);transform:translateY(-1px)}
.mc .val{font-size:1.8rem;font-weight:700}
.mc .lbl{font-size:0.75rem;color:#94a3b8;margin-top:2px}
.mc.red .val{color:#e94560}.mc.amber .val{color:#f59e0b}
.mc.green .val{color:#10b981}.mc.blue .val{color:#3b82f6}.mc.white .val{color:#1e293b}
.sec-header{font-size:1rem;font-weight:600;color:#1e293b;
  border-bottom:2px solid #f1f5f9;padding-bottom:0.4rem;margin:1.2rem 0 0.8rem}
.bubble-u{background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px 12px 4px 12px;
  padding:0.7rem 1rem;margin:0.5rem 0;max-width:80%;margin-left:auto}
.bubble-u .who{font-size:0.72rem;color:#3b82f6;margin-bottom:3px;font-weight:600}
.bubble-b{background:#fff7f7;border:1px solid #fecaca;border-radius:12px 12px 12px 4px;
  padding:0.7rem 1rem;margin:0.5rem 0;max-width:90%;
  box-shadow:0 1px 4px rgba(233,69,96,0.08)}
.bubble-b .who{font-size:0.72rem;color:#e94560;margin-bottom:3px;font-weight:600}
.bubble-b .txt{font-size:0.9rem;line-height:1.6;white-space:pre-wrap;color:#1e293b}
.ctx-box{background:#f8fafc;border-left:3px solid #3b82f6;
  padding:0.6rem;border-radius:4px;font-size:0.75rem;color:#64748b}
.upload-zone{background:#f8fafc;border:2px dashed #cbd5e1;border-radius:10px;
  padding:1.5rem;text-align:center;color:#94a3b8;margin-bottom:1rem}
.stButton>button{background:linear-gradient(135deg,#e94560,#c0392b)!important;
  color:white!important;border:none!important;border-radius:8px!important;
  font-weight:600!important;box-shadow:0 2px 8px rgba(233,69,96,0.3)!important}
.stButton>button:hover{opacity:0.9!important;transform:translateY(-1px)!important}
.stTabs [data-baseweb="tab"]{color:#94a3b8!important}
.stTabs [aria-selected="true"]{color:#e94560!important;border-bottom:2px solid #e94560!important}
.stTextInput>div>div>input{background:#ffffff!important;color:#1e293b!important;
  border:1px solid #e2e8f0!important;border-radius:8px!important}
.stTextInput>div>div>input:focus{border-color:#e94560!important;
  box-shadow:0 0 0 3px rgba(233,69,96,0.1)!important}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#f1f5f9}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:10px}
::-webkit-scrollbar-thumb:hover{background:#e94560}
</style>""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_retriever():
    try:    return Retriever(), None
    except Exception as e: return None, str(e)

def load_df(uploaded_files=None):
    frames = []
    if uploaded_files:
        for f in uploaded_files:
            try: frames.append(pd.read_csv(f))
            except: pass
    if not frames:
        try: frames.append(pd.read_csv("data/sensor_logs.csv"))
        except: pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def color_status(val):
    if val=="critical": return "background-color:#3d0000;color:#ff4757"
    if val=="warning":  return "background-color:#3d2000;color:#ffa502"
    return "background-color:#003d00;color:#3fb950"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fr-header">
  <div class="logo">🎯</div>
  <div>
    <h1>FailureRadar</h1>
    <p><span class="fr-badge">RAG</span><span class="fr-badge">Isolation Forest</span>
       <span class="fr-badge">Ollama LLM</span><span class="fr-badge">FAISS</span>
       Industrial Sensor Diagnostics Platform — v2</p>
  </div>
</div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 FailureRadar v2")
    st.markdown("---")
    st.markdown("#### 📂 Upload Sensor CSV Files")
    uploaded_files = st.file_uploader("Drop CSV files here", type=["csv"],
                                       accept_multiple_files=True)
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) loaded")

    st.markdown("---")
    st.markdown("#### ⚙️ LLM Settings")
    ollama_model = st.selectbox("Ollama Model", ["llama3.2","mistral","gemma2","llama3.1","phi3"])
    top_k        = st.slider("Retrieved Chunks", 3, 10, 6)
    show_context = st.checkbox("Show RAG Context", False)

    st.markdown("---")
    st.markdown("#### 🤖 Anomaly Detection")
    contamination = st.slider("Expected Anomaly %", 5, 30, 10) / 100
    run_ml = st.button("🔍 Run Isolation Forest", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Quick Queries")
    for q in ["Why did MOTOR-01 overheat?","Which machine had most critical events?",
               "What caused PUMP-05 bearing fault?","Summarize anomalies for COMPRESSOR-03",
               "Which equipment needs urgent maintenance?","Compare vibration across machines"]:
        if st.button(q, key=f"q_{q[:15]}"):
            st.session_state["prefill_query"] = q

    st.markdown("---")
    st.code("ollama serve\nollama pull llama3.2\npython src/ingest.py\nstreamlit run app.py",
            language="bash")


# ── Data & ML ─────────────────────────────────────────────────────────────────
df = load_df(uploaded_files if uploaded_files else None)
retriever, retriever_err = load_retriever()

if "df_analyzed" not in st.session_state:
    st.session_state.df_analyzed = pd.DataFrame()

if run_ml and not df.empty:
    with st.spinner("🤖 Running Isolation Forest on sensor features..."):
        st.session_state.df_analyzed = run_anomaly_detection(df, contamination)
    st.success("✅ Anomaly detection complete! Go to 🤖 Anomaly Detection tab.")

df_analyzed = st.session_state.df_analyzed if not st.session_state.df_analyzed.empty else df


# ── KPI Row ───────────────────────────────────────────────────────────────────
if not df.empty:
    m  = df["machine_id"].nunique()
    t  = len(df)
    cr = len(df[df["status"]=="critical"])
    wa = len(df[df["status"]=="warning"])
    no = len(df[df["status"]=="normal"])
    ac = len(df_analyzed[df_analyzed["anomaly_flag"]==-1]) if "anomaly_flag" in df_analyzed.columns else "—"
    st.markdown(f"""<div class="metric-row">
      <div class="mc blue" ><div class="val">{m}</div> <div class="lbl">🏭 Machines</div></div>
      <div class="mc white"><div class="val">{t}</div> <div class="lbl">📊 Readings</div></div>
      <div class="mc red"  ><div class="val">{cr}</div><div class="lbl">🚨 Critical</div></div>
      <div class="mc amber"><div class="val">{wa}</div><div class="lbl">⚠️ Warnings</div></div>
      <div class="mc green"><div class="val">{no}</div><div class="lbl">✅ Normal</div></div>
      <div class="mc red"  ><div class="val">{ac}</div><div class="lbl">🤖 ML Anomalies</div></div>
    </div>""", unsafe_allow_html=True)


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sensor Dashboard", "🤖 Anomaly Detection", "💬 AI Diagnostics", "📄 Export Report"
])

COLORS = ["#e94560","#58a6ff","#3fb950","#ffa502","#a371f7"]
LAYOUT = dict(paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
              legend=dict(bgcolor="#161b22"), margin=dict(l=10,r=10,t=10,b=10))

# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────
with tab1:
    if df.empty:
        st.info("Upload a CSV from the sidebar or place your file at data/sensor_logs.csv")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-header">🌡️ Temperature Over Time</div>', unsafe_allow_html=True)
            fig = px.line(df, x="timestamp", y="temperature_C", color="machine_id",
                          template="plotly_dark", color_discrete_sequence=COLORS)
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown('<div class="sec-header">📳 Vibration Distribution by Machine</div>', unsafe_allow_html=True)
            fig2 = px.box(df, x="machine_id", y="vibration_mm_s", color="machine_id",
                          template="plotly_dark", color_discrete_sequence=COLORS)
            fig2.update_layout(**{**LAYOUT, "showlegend": False})
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec-header">📋 Sensor Log</div>', unsafe_allow_html=True)
        t1, t2 = st.tabs(["All Readings", "🚨 Alerts Only"])
        with t1:
            st.dataframe(df.tail(30).style.applymap(color_status, subset=["status"]),
                         use_container_width=True, height=260)
        with t2:
            st.dataframe(df[df["alert"]!="none"].style.applymap(color_status, subset=["status"]),
                         use_container_width=True, height=260)


# ── Tab 2: Anomaly Detection ──────────────────────────────────────────────────
with tab2:
    if "anomaly_flag" not in df_analyzed.columns:
        st.markdown("""<div class="upload-zone">
            <h3>🤖 Isolation Forest Anomaly Detection</h3>
            <p>Click <b>"Run Isolation Forest"</b> in the sidebar to detect ML-based anomalies.</p>
            <p style="font-size:0.85rem;color:#58a6ff">
              Features used: Temperature · Vibration · Pressure · RPM · Current
            </p></div>""", unsafe_allow_html=True)
    else:
        anomaly_summary = get_anomaly_summary(df_analyzed)
        st.session_state["anomaly_summary"] = anomaly_summary

        cols = st.columns(len(anomaly_summary))
        for col, (machine, stats) in zip(cols, anomaly_summary.items()):
            with col:
                rate  = stats["anomaly_rate"]
                color = "#ff4757" if rate>20 else "#ffa502" if rate>10 else "#3fb950"
                st.markdown(f"""<div class="mc" style="border-color:{color}">
                  <div class="val" style="color:{color}">{rate}%</div>
                  <div class="lbl">{machine}</div>
                  <div style="font-size:0.7rem;color:#8b949e;margin-top:4px">
                    {stats['anomalies']}/{stats['total']} anomalies
                  </div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header">🎯 Anomaly Scatter — Temperature vs Vibration</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.scatter(df_analyzed, x="temperature_C", y="vibration_mm_s",
                              color="anomaly_label", template="plotly_dark",
                              color_discrete_map={"🔴 ANOMALY":"#ff4757","🟢 Normal":"#3fb950"},
                              hover_data=["machine_id","timestamp","anomaly_score"])
            fig3.update_layout(**LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.histogram(df_analyzed, x="anomaly_score", color="machine_id",
                                nbins=20, template="plotly_dark", color_discrete_sequence=COLORS,
                                labels={"anomaly_score":"Anomaly Score"})
            fig4.update_layout(**LAYOUT)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="sec-header">🔴 Flagged Anomaly Records</div>', unsafe_allow_html=True)
        anom_df = df_analyzed[df_analyzed["anomaly_flag"]==True][[
            "timestamp","machine_id","temperature_C","vibration_mm_s",
            "pressure_bar","status","anomaly_score","anomaly_label"
        ]].sort_values("anomaly_score", ascending=False)
        st.dataframe(anom_df, use_container_width=True, height=280)


# ── Tab 3: AI Diagnostics ─────────────────────────────────────────────────────
with tab3:
    if retriever_err:
        st.warning(f"⚠️ Vector store not found. Run: `python src/ingest.py`\n\n_{retriever_err}_")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""<div class="bubble-u">
              <div class="who">👷 Engineer</div>
              <div class="txt">{msg['content']}</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="bubble-b">
              <div class="who">🎯 FailureRadar AI</div>
              <div class="txt">{msg['content']}</div></div>""", unsafe_allow_html=True)
            if show_context and "context" in msg:
                with st.expander("📎 RAG Context"):
                    st.markdown(f'<div class="ctx-box">{msg["context"]}</div>',
                                unsafe_allow_html=True)

    prefill    = st.session_state.pop("prefill_query", "")
    user_input = st.text_input("Ask about your equipment...", value=prefill,
                               placeholder="e.g. Why did MOTOR-01 overheat?")
    c1, c2 = st.columns([5,1])
    with c1: send = st.button("🔍 Analyze", use_container_width=True)
    with c2:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            st.rerun()

    if send and user_input.strip() and retriever:
        q = user_input.strip()
        st.session_state.messages.append({"role":"user","content":q})
        with st.spinner("🔎 Retrieving sensor context... 🧠 Generating diagnosis..."):
            docs    = retriever.retrieve(q, top_k=top_k)
            context = retriever.format_context(docs)
            prompt  = build_rag_prompt(q, context)
            answer  = ask_ollama(prompt, model=ollama_model)
        st.session_state.messages.append({
            "role":"assistant","content":answer,
            "context":context.replace("\n","<br>")
        })
        st.rerun()
    elif send and not retriever:
        st.error("Run `python src/ingest.py` first to build the vector store.")


# ── Tab 4: Export ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-header">📄 Export Full Diagnostic Report</div>',
                unsafe_allow_html=True)
    st.markdown("Generate a professional report containing fleet metrics, ML anomaly summary, and AI Q&A.")

    anomaly_summary = st.session_state.get("anomaly_summary", {})
    messages        = st.session_state.get("messages", [])

    if not messages and not anomaly_summary:
        st.info("💡 Run anomaly detection and ask some questions first, then come back here to export.")

    c1, c2 = st.columns(2)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    with c1:
        st.markdown("#### 📕 PDF Report")
        st.markdown("Formatted report with tables and colour-coded sections.")
        if st.button("⬇️ Generate PDF", use_container_width=True):
            with st.spinner("Building PDF..."):
                pdf_bytes = generate_pdf_report(messages, df, df_analyzed if "anomaly_flag" in df_analyzed.columns else None)
            if pdf_bytes:
                st.download_button("💾 Download PDF", data=pdf_bytes,
                    file_name=f"FailureRadar_{ts}.pdf", mime="application/pdf",
                    use_container_width=True)
            else:
                st.error("Install reportlab: `pip install reportlab`")

    with c2:
        st.markdown("#### 📘 Word Document")
        st.markdown("Editable DOCX report you can share with your team.")
        if st.button("⬇️ Generate DOCX", use_container_width=True):
            with st.spinner("Building DOCX..."):
                docx_bytes = generate_docx_report(messages, df, df_analyzed if "anomaly_flag" in df_analyzed.columns else None)
            if docx_bytes:
                st.download_button("💾 Download DOCX", data=docx_bytes,
                    file_name=f"FailureRadar_{ts}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True)
            else:
                st.error("Install python-docx: `pip install python-docx`")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""<div style="text-align:center;color:#cbd5e1;font-size:0.78rem;
  margin-top:2rem;padding-top:1rem;border-top:1px solid #e2e8f0">
  🎯 FailureRadar v2 · RAG + Isolation Forest + Ollama + FAISS · Built by Puneet Divedi
</div>""", unsafe_allow_html=True)