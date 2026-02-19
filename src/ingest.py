"""
ingest.py — Load sensor CSV logs, convert to text chunks, embed with
sentence-transformers, and persist a FAISS index to disk.

Run once (or whenever data updates):
    python src/ingest.py
"""

import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH      = "data/sensor_logs.csv"
VECTORSTORE    = "vectorstore/index.faiss"
METADATA_PATH  = "vectorstore/metadata.pkl"
EMBED_MODEL    = "all-MiniLM-L6-v2"   # fast & free, runs locally


def row_to_text(row: pd.Series) -> str:
    """Convert a sensor reading row into a natural-language string for embedding."""
    alert_msg = f" ALERT: {row['alert']}." if row['alert'] != 'none' else ""
    return (
        f"Machine {row['machine_id']} at {row['timestamp']}: "
        f"Temperature={row['temperature_C']}°C, "
        f"Vibration={row['vibration_mm_s']} mm/s, "
        f"Pressure={row['pressure_bar']} bar, "
        f"RPM={row['rpm']}, "
        f"Current={row['current_A']} A, "
        f"Status={row['status'].upper()}.{alert_msg}"
    )


def build_summary_chunks(df: pd.DataFrame) -> list[dict]:
    """Build machine-level summary chunks in addition to row-level chunks."""
    chunks = []
    for machine_id, grp in df.groupby("machine_id"):
        criticals = grp[grp["status"] == "critical"]
        warnings  = grp[grp["status"] == "warning"]
        alerts    = grp[grp["alert"] != "none"]["alert"].value_counts()

        summary = (
            f"Summary for {machine_id}: "
            f"Total readings={len(grp)}, "
            f"Critical events={len(criticals)}, "
            f"Warning events={len(warnings)}, "
            f"Max temperature={grp['temperature_C'].max()}°C, "
            f"Max vibration={grp['vibration_mm_s'].max()} mm/s, "
            f"Min pressure={grp['pressure_bar'].min()} bar. "
            f"Alert types: {dict(alerts) if not alerts.empty else 'none'}."
        )
        chunks.append({"text": summary, "source": f"{machine_id}_summary", "type": "summary"})
    return chunks


def ingest():
    os.makedirs("vectorstore", exist_ok=True)

    print(f"📂  Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"    {len(df)} rows loaded.")

    # ── Build text chunks ────────────────────────────────────────────────────
    row_chunks = [
        {"text": row_to_text(row), "source": f"row_{i}", "type": "reading"}
        for i, (_, row) in enumerate(df.iterrows())
    ]
    summary_chunks = build_summary_chunks(df)
    all_chunks = row_chunks + summary_chunks

    texts = [c["text"] for c in all_chunks]
    print(f"    {len(texts)} chunks created ({len(row_chunks)} readings + {len(summary_chunks)} summaries).")

    # ── Embed ────────────────────────────────────────────────────────────────
    print(f"\n🔢  Embedding with '{EMBED_MODEL}'...")
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # ── FAISS index ──────────────────────────────────────────────────────────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product on normalized vecs = cosine sim
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, VECTORSTORE)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n✅  FAISS index saved → {VECTORSTORE}")
    print(f"✅  Metadata saved   → {METADATA_PATH}")
    print(f"    Index size: {index.ntotal} vectors, dim={dim}")


if __name__ == "__main__":
    ingest()
