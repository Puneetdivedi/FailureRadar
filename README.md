# 🏭 IndustrialCopilot — RAG-Powered Sensor Diagnostics Assistant

> **Ask natural language questions about your plant equipment and get LLM-powered root cause analysis — grounded in real sensor data.**

![Tech Stack](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama3.2-blue)
![RAG](https://img.shields.io/badge/RAG-FAISS%20%7C%20SentenceTransformers-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

---

## 🎯 What It Does

IndustrialCopilot lets manufacturing engineers ask plain-English questions about sensor data and get intelligent, grounded diagnostic responses from a local LLM — **no cloud API needed, no data leaves your machine.**

**Example queries:**
- *"Why did MOTOR-01 overheat on January 1st?"*
- *"Which machine had the most critical events?"*
- *"What caused the PUMP-05 bearing fault?"*
- *"Recommend maintenance actions for COMPRESSOR-03."*

---

## 🏗️ Architecture

```
CSV Sensor Logs
      │
      ▼
┌─────────────────┐
│   ingest.py     │  → Row-level + summary text chunks
│  (Chunking)     │
└────────┬────────┘
         │ SentenceTransformers (all-MiniLM-L6-v2)
         ▼
┌─────────────────┐
│  FAISS Index    │  ← Cosine similarity search (IndexFlatIP)
│  (vectorstore/) │
└────────┬────────┘
         │ Top-K retrieval
         ▼
┌─────────────────┐     ┌──────────────────┐
│  retriever.py   │────▶│   RAG Prompt     │
│ (Semantic Search│     │  (context + Q)   │
└─────────────────┘     └────────┬─────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │   Ollama LLM     │  (llama3.2 / mistral / gemma2)
                        │   (llm.py)       │  runs 100% locally
                        └────────┬─────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  Streamlit UI    │  (app.py)
                        │  Dashboard +     │
                        │  Chat Interface  │
                        └──────────────────┘
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install & Start Ollama
```bash
# Download Ollama from https://ollama.ai
ollama serve
ollama pull llama3.2    # or: mistral, gemma2, phi3
```

### 3. Build the Vector Store
```bash
python src/ingest.py
```

### 4. Launch the App
```bash
streamlit run app.py
```

Open http://localhost:8501 🚀

---

## 📁 Project Structure

```
industrial-rag/
├── app.py                  # Streamlit UI (dashboard + chat)
├── requirements.txt
├── data/
│   └── sensor_logs.csv     # Industrial sensor data (plug in your real data)
├── src/
│   ├── ingest.py           # CSV → text chunks → FAISS index
│   ├── retriever.py        # Semantic search over FAISS
│   └── llm.py              # Ollama API interface + RAG prompt builder
└── vectorstore/
    ├── index.faiss         # FAISS vector index (auto-generated)
    └── metadata.pkl        # Chunk metadata (auto-generated)
```

---

## 🔧 Plug In Your Own Data

Replace `data/sensor_logs.csv` with your real sensor export. Expected columns:

| Column | Description |
|---|---|
| `timestamp` | ISO datetime |
| `machine_id` | Equipment identifier |
| `temperature_C` | Temperature reading |
| `vibration_mm_s` | Vibration in mm/s |
| `pressure_bar` | Pressure in bar |
| `rpm` | Rotations per minute |
| `current_A` | Electrical current |
| `status` | `normal` / `warning` / `critical` |
| `alert` | Alert type or `none` |

Then re-run `python src/ingest.py` to rebuild the index.

---

## 🧠 Key Technical Decisions

| Component | Choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | Fast, free, local, great for semantic similarity |
| Vector DB | FAISS `IndexFlatIP` | Exact cosine search, no server needed |
| LLM | Ollama (local) | 100% private, no API costs |
| Chunking | Row-level + machine summaries | Enables both specific and aggregate queries |
| RCA Engine | Rule-based Heuristics | Fast, explainable diagnoses of detected anomalies |
| Temperature | 0.2 | Low = factual, deterministic diagnostics |

---

## 📈 Resume Highlights

- **RAG pipeline** from raw CSV to FAISS vector store with semantic retrieval
- **Local LLM integration** via Ollama (privacy-first, zero API cost)
- **Dual-chunk strategy**: individual readings + machine-level summaries
- **Production UI** with real-time sensor dashboard + conversational chat
- Applied to real **industrial/manufacturing** domain (Bosch-relevant)

---

## 🚀 Possible Extensions

- [x] Add Heuristic Root Cause Analysis (RCA)
- [ ] Add LangChain for multi-step agent reasoning
- [ ] Integrate MLflow anomaly detection model for hybrid ML+RAG
- [ ] Export auto-generated RCA reports as PDF/DOCX
- [ ] Add real-time CSV streaming with watchdog
- [ ] Multi-file ingestion (maintenance manuals + sensor logs)

---

*Built by Puneet Divedi · GenAI Engineer · RAG | LLMs | Industrial AI*
