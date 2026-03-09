"""
Index Wikipedia passages into Qdrant (dense) + BM25 (sparse).

Usage:
    python rag/index.py                     # index only pages referenced in train.jsonl
    python rag/index.py --all               # index all main-namespace pages (very slow)
    python rag/index.py --limit 1000        # quick test with 1000 pages
"""

import json
import re
import pickle
import sqlite3
import argparse
from pathlib import Path

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "wikipedia_pages.sqlite"
TRAIN_PATH = DATA_DIR / "train.jsonl"
INDEX_DIR = DATA_DIR / "index"

COLLECTION = "wiki_passages"
EMBED_MODEL = "jinaai/jina-embeddings-v3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_BATCH = 64


# ── wikitext cleanup ──────────────────────────────────────────────

def clean_wikitext(raw: str) -> str:
    if not raw:
        return ""
    t = raw
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.DOTALL)
    t = re.sub(r"<ref[^/]*/>", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\{\{[^}]*\}\}", "", t)
    t = re.sub(r"\[\[[^\]]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S*\s?([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"={2,}(.+?)={2,}", r"\1", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


# ── chunking ──────────────────────────────────────────────────────

def chunk_text(text: str, title: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    pos = 0
    while pos < len(text):
        piece = text[pos : pos + size].strip()
        if piece:
            chunks.append(f"{title}\n{piece}")
        pos += size - overlap
    return chunks


# ── data loading ──────────────────────────────────────────────────

def get_relevant_ids(path: str) -> set[str]:
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for wid in json.loads(line).get("wikipedia_id", []):
                ids.add(str(wid))
    return ids


def load_pages(db_path: str, ids: set[str] | None = None) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if ids:
        rows = []
        batch = list(ids)
        STEP = 500  # sqlite variable limit
        for i in range(0, len(batch), STEP):
            sub = batch[i : i + STEP]
            ph = ",".join("?" * len(sub))
            rows.extend(
                conn.execute(
                    f"SELECT wikipedia_id, title, text FROM pages WHERE wikipedia_id IN ({ph})",
                    sub,
                ).fetchall()
            )
    else:
        rows = conn.execute(
            "SELECT wikipedia_id, title, text FROM pages WHERE ns = 0"
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


# ── main build ────────────────────────────────────────────────────

def build_index(
    pages: list[dict],
    index_dir: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_batch: int = EMBED_BATCH,
):
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1) chunk
    all_chunks: list[str] = []
    meta: list[dict] = []
    for p in tqdm(pages, desc="chunking"):
        text = clean_wikitext(p["text"] or "")
        for i, c in enumerate(chunk_text(text, p["title"], chunk_size, chunk_overlap)):
            all_chunks.append(c)
            meta.append(
                {
                    "wikipedia_id": p["wikipedia_id"],
                    "title": p["title"],
                    "chunk_idx": i,
                    "text": c,
                }
            )
    print(f"total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("nothing to index")
        return

    # 2) embed with jina-v3
    print(f"loading {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

    print("encoding passages ...")
    vectors = model.encode(
        all_chunks,
        batch_size=embed_batch,
        show_progress_bar=True,
        normalize_embeddings=True,
        task="retrieval.passage",
    )
    dim = vectors.shape[1]
    print(f"embedding dim: {dim}")

    # 3) qdrant — local file storage, no server needed
    qpath = str(index_dir / "qdrant_store")
    client = QdrantClient(path=qpath)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    UPLOAD_BATCH = 512
    for i in tqdm(range(0, len(all_chunks), UPLOAD_BATCH), desc="qdrant upload"):
        pts = [
            PointStruct(id=j, vector=vectors[j].tolist(), payload=meta[j])
            for j in range(i, min(i + UPLOAD_BATCH, len(all_chunks)))
        ]
        client.upsert(COLLECTION, pts)

    client.close()
    print(f"qdrant: {len(all_chunks)} points in '{COLLECTION}'")

    # 4) bm25
    print("building bm25 ...")
    tokenized = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    bm25_path = index_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "meta": meta}, f)
    print(f"bm25 saved → {bm25_path}")

    # 5) save config for the RAG part
    config = {
        "embed_model": EMBED_MODEL,
        "collection": COLLECTION,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "dim": dim,
        "n_chunks": len(all_chunks),
        "n_pages": len(pages),
    }
    with open(index_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("done ✓")


# ── CLI ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build Qdrant + BM25 index over Wikipedia passages")
    ap.add_argument("--all", action="store_true", help="index ALL main-namespace pages (millions, very slow)")
    ap.add_argument("--limit", type=int, default=100, help="max pages to index (non-positive = no limit)")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--train", default=str(TRAIN_PATH))
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--embed-batch", type=int, default=EMBED_BATCH)
    args = ap.parse_args()

    if args.all:
        print("loading ALL main-namespace pages ...")
        pages = load_pages(args.db)
    else:
        ids = get_relevant_ids(args.train)
        print(f"found {len(ids)} relevant page IDs in train.jsonl")
        pages = load_pages(args.db, ids)

    if args.limit > 0:
        pages = pages[: args.limit]

    print(f"pages to index: {len(pages)}")

    build_index(
        pages,
        Path(args.index_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_batch=args.embed_batch,
    )


if __name__ == "__main__":
    main()
