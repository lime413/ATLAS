"""
Stream Wikipedia passages into a hybrid index:
- dense vectors in local Qdrant
- sparse text search in SQLite FTS5

This version avoids loading all pages/chunks/BM25 structures into memory.
"""

import argparse
import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "wikipedia_pages_50k.sqlite"
INDEX_DIR = DATA_DIR / "index_50k"

COLLECTION = "wiki_passages"
EMBED_MODEL = "jinaai/jina-embeddings-v3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PAGE_BATCH = 128
EMBED_BATCH = 64
FLUSH_CHUNKS = 2048
CLEAN_MODE = "raw-ish"


def normalize_wiki_formatting(text: str) -> str:
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"={2,}(.+?)={2,}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_wikitext(raw: str, mode: str = CLEAN_MODE) -> str:
    if not raw:
        return ""

    text = raw
    if mode == "light-clean":
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
        text = re.sub(r"<ref[^/]*/>", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\[\[[^\]]*\|([^\]]*)\]\]", r"\1", text)
        text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
        text = re.sub(r"\[https?://\S*\s?([^\]]*)\]", r"\1", text)
    elif mode != "raw-ish":
        raise ValueError(f"Unsupported clean mode: {mode}")

    return normalize_wiki_formatting(text)


def chunk_text(text: str, title: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []

    chunks = []
    pos = 0
    stride = max(1, size - overlap)
    while pos < len(text):
        piece = text[pos : pos + size].strip()
        if piece:
            chunks.append(f"{title}\n{piece}")
        pos += stride
    return chunks


def reset_index_dir(index_dir: Path) -> None:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)


def init_passages_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA mmap_size=1073741824;")
    conn.execute(
        """
        CREATE TABLE passages (
            chunk_id      INTEGER PRIMARY KEY,
            wikipedia_id  TEXT NOT NULL,
            title         TEXT NOT NULL,
            chunk_idx     INTEGER NOT NULL,
            text          TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE passages_fts USING fts5(
            title,
            text,
            content='passages',
            content_rowid='chunk_id'
        )
        """
    )
    conn.execute("CREATE INDEX idx_passages_wikipedia_id ON passages(wikipedia_id)")
    conn.commit()
    return conn


def count_pages(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    conn.close()
    return total


def iter_pages(db_path: Path, page_limit: int | None, page_batch: int) -> Iterable[list[sqlite3.Row]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = "SELECT wikipedia_id, title, text FROM pages ORDER BY wikipedia_id"
    params: tuple[int, ...] = ()
    if page_limit is not None and page_limit > 0:
        query += " LIMIT ?"
        params = (page_limit,)

    cur.execute(query, params)
    try:
        while True:
            rows = cur.fetchmany(page_batch)
            if not rows:
                break
            yield rows
    finally:
        conn.close()


def flush_chunks(
    chunk_rows: list[dict],
    embedder: SentenceTransformer,
    qdrant: QdrantClient,
    passages_conn: sqlite3.Connection,
    collection: str,
    embed_batch: int,
    vector_dim: int | None,
) -> tuple[int | None, int]:
    if not chunk_rows:
        return vector_dim, 0

    texts = [row["text"] for row in chunk_rows]
    vectors = embedder.encode(
        texts,
        batch_size=embed_batch,
        show_progress_bar=False,
        normalize_embeddings=True,
        task="retrieval.passage",
        convert_to_numpy=True,
    )

    if vector_dim is None:
        vector_dim = int(vectors.shape[1])
        if qdrant.collection_exists(collection):
            qdrant.delete_collection(collection)
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    points = [
        PointStruct(id=row["chunk_id"], vector=vectors[idx].tolist())
        for idx, row in enumerate(chunk_rows)
    ]
    qdrant.upsert(collection_name=collection, points=points)

    passages_cur = passages_conn.cursor()
    passages_cur.executemany(
        """
        INSERT INTO passages (chunk_id, wikipedia_id, title, chunk_idx, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                row["chunk_id"],
                row["wikipedia_id"],
                row["title"],
                row["chunk_idx"],
                row["text"],
            )
            for row in chunk_rows
        ],
    )
    passages_cur.executemany(
        "INSERT INTO passages_fts(rowid, title, text) VALUES (?, ?, ?)",
        [(row["chunk_id"], row["title"], row["text"]) for row in chunk_rows],
    )
    passages_conn.commit()
    return vector_dim, len(chunk_rows)


def build_index(
    db_path: Path,
    index_dir: Path,
    clean_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    page_batch: int,
    embed_batch: int,
    flush_chunks_at: int,
    page_limit: int | None,
    embed_model: str,
) -> None:
    reset_index_dir(index_dir)

    passages_db_path = index_dir / "passages.sqlite"
    passages_conn = init_passages_db(passages_db_path)
    qdrant = QdrantClient(path=str(index_dir / "qdrant_store"))
    embedder = SentenceTransformer(embed_model, trust_remote_code=True)

    total_pages = count_pages(db_path)
    if page_limit is not None and page_limit > 0:
        total_pages = min(total_pages, page_limit)

    chunk_rows: list[dict] = []
    vector_dim: int | None = None
    next_chunk_id = 0
    processed_pages = 0
    total_chunks = 0

    try:
        for page_rows in tqdm(iter_pages(db_path, page_limit, page_batch), total=(total_pages + page_batch - 1) // page_batch, desc="page batches"):
            for page in page_rows:
                processed_pages += 1
                cleaned = clean_wikitext(page["text"] or "", mode=clean_mode)
                for chunk_idx, chunk in enumerate(
                    chunk_text(cleaned, page["title"], size=chunk_size, overlap=chunk_overlap)
                ):
                    chunk_rows.append(
                        {
                            "chunk_id": next_chunk_id,
                            "wikipedia_id": str(page["wikipedia_id"]),
                            "title": page["title"],
                            "chunk_idx": chunk_idx,
                            "text": chunk,
                        }
                    )
                    next_chunk_id += 1

                if len(chunk_rows) >= flush_chunks_at:
                    vector_dim, uploaded = flush_chunks(
                        chunk_rows=chunk_rows,
                        embedder=embedder,
                        qdrant=qdrant,
                        passages_conn=passages_conn,
                        collection=COLLECTION,
                        embed_batch=embed_batch,
                        vector_dim=vector_dim,
                    )
                    total_chunks += uploaded
                    chunk_rows.clear()

            if processed_pages % 1000 == 0:
                print(f"processed_pages={processed_pages} | total_chunks={total_chunks}")

        vector_dim, uploaded = flush_chunks(
            chunk_rows=chunk_rows,
            embedder=embedder,
            qdrant=qdrant,
            passages_conn=passages_conn,
            collection=COLLECTION,
            embed_batch=embed_batch,
            vector_dim=vector_dim,
        )
        total_chunks += uploaded
        chunk_rows.clear()

        config = {
            "embed_model": embed_model,
            "collection": COLLECTION,
            "clean_mode": clean_mode,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "dim": vector_dim,
            "n_chunks": total_chunks,
            "n_pages": processed_pages,
            "passages_db": "passages.sqlite",
        }
        with (index_dir / "config.json").open("w", encoding="utf-8") as fout:
            json.dump(config, fout, indent=2)

        print({"pages_indexed": processed_pages, "chunks_indexed": total_chunks, "index_dir": str(index_dir)})
    finally:
        passages_conn.close()
        qdrant.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a streaming hybrid index over Wikipedia passages")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument(
        "--clean-mode",
        choices=["raw-ish", "light-clean"],
        default=CLEAN_MODE,
        help=(
            "Text cleanup level before chunking. "
            "'raw-ish' keeps pages close to raw Wikipedia text and only normalizes formatting: "
            "collapse repeated spaces, collapse extra blank lines, remove wiki bold/italic markup, "
            "and turn headings like '== History ==' into plain text. "
            "'light-clean' applies the same formatting normalization plus removes <ref>...</ref> blocks, "
            "strips HTML tags, unwraps wiki links like [[A|B]] into 'B' and [[A]] into 'A', "
            "and removes URL wrappers such as [http://... label] while keeping the label."
        ),
    )
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--page-batch", type=int, default=PAGE_BATCH)
    parser.add_argument("--embed-batch", type=int, default=EMBED_BATCH)
    parser.add_argument("--flush-chunks", type=int, default=FLUSH_CHUNKS)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    args = parser.parse_args()

    build_index(
        db_path=Path(args.db),
        index_dir=Path(args.index_dir),
        clean_mode=args.clean_mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        page_batch=args.page_batch,
        embed_batch=args.embed_batch,
        flush_chunks_at=args.flush_chunks,
        page_limit=args.page_limit,
        embed_model=args.embed_model,
    )


if __name__ == "__main__":
    main()
