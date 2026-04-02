"""
Copy vectors from embedded Qdrant (index_dir/qdrant_store) to a running Qdrant server.

Use this on Windows instead of bind-mounting NTFS into the container (unsupported / risky).

Memory: the *source* uses ``QdrantClient(path=...)`` (embedded Qdrant in-process). That can use a lot
of RAM for the on-disk index, similar to old local retrieval. Batch buffers are bounded by
``--batch-size`` (default 64); use 32 or 16 if you still spike.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Repo root on sys.path when run as python rag/migrate_qdrant_to_server.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.qdrant_backend import DEFAULT_QDRANT_URL, open_qdrant_client


def migrate(
    index_dir: Path,
    qdrant_url: str,
    batch_size: int,
    recreate: bool,
) -> None:
    with (index_dir / "config.json").open(encoding="utf-8") as fin:
        config = json.load(fin)
    collection = config["collection"]
    dim = int(config["dim"])
    batch_size = max(1, int(batch_size))

    local = QdrantClient(path=str(index_dir / "qdrant_store"))
    remote = open_qdrant_client(url=qdrant_url)
    try:
        if recreate and remote.collection_exists(collection):
            remote.delete_collection(collection)
        if not remote.collection_exists(collection):
            remote.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        offset = None
        total = 0
        while True:
            records, offset = local.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            if not records:
                break
            points = [
                PointStruct(id=r.id, vector=r.vector, payload=r.payload)
                for r in records
            ]
            n = len(points)
            remote.upsert(collection_name=collection, points=points)
            total += n
            print(f"upserted={total}")
            del records, points
            gc.collect()
            if offset is None:
                break
        print({"done": True, "points": total, "collection": collection})
    finally:
        local.close()
        remote.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate embedded Qdrant index to a Docker/server instance.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Scroll/upsert chunk size (smaller = lower peak RAM for batch buffers; embedded source still heavy).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the remote collection if it exists before uploading.",
    )
    args = parser.parse_args()
    migrate(
        index_dir=args.index_dir.resolve(),
        qdrant_url=args.qdrant_url.strip(),
        batch_size=args.batch_size,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
