"""Hybrid retrieval + RAG answer generation over a FAISS + SQLite index."""

import argparse
import gc
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

try:
    from rag.constants import DEFAULT_LLM_MODEL
    from rag.embedder_factory import make_sentence_transformer
except ImportError:
    from constants import DEFAULT_LLM_MODEL
    from embedder_factory import make_sentence_transformer

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index_50k"
LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = DEFAULT_LLM_MODEL

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_PASSAGE_FETCH_CHUNK = min(400, int(os.environ.get("ATLAS_PASSAGE_FETCH_CHUNK", "400")))


def load_index(index_dir: Path = INDEX_DIR) -> dict[str, Any]:
    with (index_dir / "config.json").open(encoding="utf-8") as fin:
        config = json.load(fin)

    dense_index_file = index_dir / config.get("dense_index_file", "faiss.index")
    if not dense_index_file.exists():
        raise FileNotFoundError(f"Dense index file not found: {dense_index_file}")

    embedder = make_sentence_transformer(config["embed_model"])
    faiss_index = faiss.read_index(str(dense_index_file))
    passages_db = (index_dir / config["passages_db"]).resolve()
    passages_conn = sqlite3.connect(f"{passages_db.as_uri()}?mode=ro", uri=True)
    passages_conn.row_factory = sqlite3.Row
    cache_kb = int(os.environ.get("ATLAS_SQLITE_CACHE_KB", "4096"))
    passages_conn.execute(f"PRAGMA cache_size={-max(512, cache_kb)}")
    mmap = int(os.environ.get("ATLAS_SQLITE_MMAP_MB", "128"))
    if mmap > 0:
        passages_conn.execute(f"PRAGMA mmap_size={mmap * 1024 * 1024}")
    else:
        passages_conn.execute("PRAGMA mmap_size=0")

    return {
        "embedder": embedder,
        "faiss_index": faiss_index,
        "passages_conn": passages_conn,
        "config": config,
    }


def close_index(index: dict[str, Any]) -> None:
    embedder = index.pop("embedder", None)
    if embedder is not None:
        del embedder
    faiss_index = index.pop("faiss_index", None)
    if faiss_index is not None:
        del faiss_index
    index["passages_conn"].close()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def to_fts_query(text: str) -> str:
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return ""
    return " OR ".join(tokens[:20])


def fetch_passages_by_ids(conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not chunk_ids:
        return {}
    seen: dict[int, None] = {}
    for cid in chunk_ids:
        seen[int(cid)] = None
    unique_ids = list(seen.keys())
    out: dict[int, dict[str, Any]] = {}
    chunk = max(1, _PASSAGE_FETCH_CHUNK)
    for i in range(0, len(unique_ids), chunk):
        batch = unique_ids[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"""
            SELECT chunk_id, wikipedia_id, title, chunk_idx, text
            FROM passages
            WHERE chunk_id IN ({placeholders})
            """,
            batch,
        ).fetchall()
        for row in rows:
            cid = int(row["chunk_id"])
            out[cid] = {
                "chunk_id": cid,
                "wikipedia_id": row["wikipedia_id"],
                "title": row["title"],
                "chunk_idx": int(row["chunk_idx"]),
                "text": row["text"],
            }
    return out


def dense_ids_from_vector(index: dict[str, Any], vector: Any, limit: int) -> list[int]:
    q = np.asarray(vector, dtype="float32")
    if q.ndim == 2:
        raise ValueError("pass a single query vector (1D), not a batch matrix")
    q = np.ascontiguousarray(q.reshape(1, -1), dtype="float32")
    _, hits = index["faiss_index"].search(q, limit)
    return [int(chunk_id) for chunk_id in hits[0].tolist() if int(chunk_id) >= 0]


def dense_retrieve(query: str, index: dict[str, Any], limit: int) -> list[int]:
    import torch

    with torch.inference_mode():
        q_vec = index["embedder"].encode(
            query,
            task="retrieval.query",
            normalize_embeddings=True,
        )
    return dense_ids_from_vector(index, q_vec, limit)


def sparse_retrieve(query: str, index: dict[str, Any], limit: int) -> list[int]:
    fts_query = to_fts_query(query)
    if not fts_query:
        return []

    rows = index["passages_conn"].execute(
        """
        SELECT passages.chunk_id
        FROM passages_fts
        JOIN passages ON passages.chunk_id = passages_fts.rowid
        WHERE passages_fts MATCH ?
        ORDER BY bm25(passages_fts)
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    return [int(row["chunk_id"]) for row in rows]


def retrieve(
    query: str,
    index: dict[str, Any],
    top_k: int = 5,
    dense_limit: int = 20,
    sparse_limit: int = 20,
    sparse_weight: float = 0.3,
    *,
    query_embedding: Any | None = None,
) -> list[dict[str, Any]]:
    if query_embedding is None:
        dense_ids = dense_retrieve(query, index, limit=dense_limit)
    else:
        dense_ids = dense_ids_from_vector(index, query_embedding, limit=dense_limit)
    sparse_ids = sparse_retrieve(query, index, limit=sparse_limit)

    rrf: dict[int, float] = {}
    for rank, chunk_id in enumerate(dense_ids):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + (1.0 - sparse_weight) / (rank + 60)
    for rank, chunk_id in enumerate(sparse_ids):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + sparse_weight / (rank + 60)

    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
    passages = fetch_passages_by_ids(index["passages_conn"], top_ids)
    return [passages[chunk_id] for chunk_id in top_ids if chunk_id in passages]


def rag_answer(
    query: str,
    index: dict[str, Any],
    top_k: int = 5,
    dense_limit: int = 20,
    sparse_limit: int = 20,
    sparse_weight: float = 0.3,
    max_tokens: int = 256,
    base_url: str = LLM_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> tuple[str, list[dict[str, Any]]]:
    passages = retrieve(
        query=query,
        index=index,
        top_k=top_k,
        dense_limit=dense_limit,
        sparse_limit=sparse_limit,
        sparse_weight=sparse_weight,
    )

    context = "\n\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    prompt = (
        "Answer the question using ONLY the provided context. Be concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    client = OpenAI(base_url=base_url, api_key="not-needed")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip(), passages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval and optional RAG answer generation")
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--question", default="what do the 3 dots mean in math")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-limit", type=int, default=20)
    parser.add_argument("--sparse-limit", type=int, default=20)
    parser.add_argument("--sparse-weight", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-url", default=LLM_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    index = load_index(Path(args.index_dir))
    try:
        if args.retrieve_only:
            passages = retrieve(
                query=args.question,
                index=index,
                top_k=args.top_k,
                dense_limit=args.dense_limit,
                sparse_limit=args.sparse_limit,
                sparse_weight=args.sparse_weight,
            )
            print(f"Q: {args.question}")
            print("\nSources:")
            for passage in passages:
                print(f"- [{passage['title']}] chunk {passage['chunk_idx']}")
        else:
            answer, passages = rag_answer(
                query=args.question,
                index=index,
                top_k=args.top_k,
                dense_limit=args.dense_limit,
                sparse_limit=args.sparse_limit,
                sparse_weight=args.sparse_weight,
                max_tokens=args.max_tokens,
                base_url=args.base_url,
                model=args.model,
            )
            print(f"Q: {args.question}")
            print(f"A: {answer}")
            print("\nSources:")
            for passage in passages:
                print(f"- [{passage['title']}] chunk {passage['chunk_idx']}")
    finally:
        close_index(index)


if __name__ == "__main__":
    main()