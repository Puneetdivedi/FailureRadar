"""
retriever.py — Semantic retrieval from the FAISS vector store.

Usage:
    from src.retriever import Retriever
    r = Retriever()
    docs = r.retrieve("Why did MOTOR-01 overheat?", top_k=5)
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

VECTORSTORE   = "vectorstore/index.faiss"
METADATA_PATH = "vectorstore/metadata.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        print("🔍  Loading FAISS index and embedder...")
        self.index    = faiss.read_index(VECTORSTORE)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"    Index ready: {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
        """Return top_k most relevant chunks for the query."""
        q_vec = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(q_vec, dtype="float32"), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def format_context(self, docs: list[dict]) -> str:
        """Format retrieved docs into a clean context string for the LLM prompt."""
        lines = []
        for i, doc in enumerate(docs, 1):
            lines.append(f"[{i}] ({doc['source']}) {doc['text']}")
        return "\n".join(lines)
