"""
RAG inference: hybrid retrieval (Qdrant + BM25) → LLM answer via llama.cpp.
"""

import json
import pickle
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"

LLM_BASE_URL = "http://127.0.0.1:8080/v1"


def load_index(index_dir: Path = INDEX_DIR):
    config = json.load(open(index_dir / "config.json"))

    embedder = SentenceTransformer(config["embed_model"], trust_remote_code=True)
    qdrant = QdrantClient(path=str(index_dir / "qdrant_store"))

    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    return {
        "embedder": embedder,
        "qdrant": qdrant,
        "collection": config["collection"],
        "bm25": bm25_data["bm25"],
        "bm25_meta": bm25_data["meta"],
    }


def retrieve(query: str, index, top_k=5, bm25_weight=0.3):
    """Hybrid retrieval: dense (Qdrant) + sparse (BM25), fused by RRF."""

    # dense
    q_vec = index["embedder"].encode(
        query, task="retrieval.query", normalize_embeddings=True
    )
    dense_hits = index["qdrant"].query_points(
        collection_name=index["collection"],
        query=q_vec.tolist(),
        limit=top_k * 2,
    ).points

    # sparse
    tokens = query.lower().split()
    bm25_scores = index["bm25"].get_scores(tokens)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: top_k * 2]

    # reciprocal rank fusion
    rrf = {}
    for rank, hit in enumerate(dense_hits):
        rrf[hit.id] = rrf.get(hit.id, 0) + (1 - bm25_weight) / (rank + 60)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] = rrf.get(idx, 0) + bm25_weight / (rank + 60)

    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

    meta = index["bm25_meta"]
    return [meta[i] for i in top_ids]


def rag_answer(query: str, index, top_k=5, max_tokens=256):
    passages = retrieve(query, index, top_k=top_k)

    context = "\n\n".join(
        f"[{i+1}] {p['text']}" for i, p in enumerate(passages)
    )

    prompt = (
        f"Answer the question using ONLY the provided context. "
        f"Be concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    print('full prompt:', prompt)
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    resp = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content, passages


if __name__ == "__main__":
    idx = load_index()

    question = "what do the 3 dots mean in math"
    answer, sources = rag_answer(question, idx)

    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"\nSources:")
    for s in sources:
        print(f"  - [{s['title']}] chunk {s['chunk_idx']}")
