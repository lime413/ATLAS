"""
Build a smaller Wikipedia SQLite + train_hard JSONL subset:

1. Collect all wikipedia_id values referenced in train_hard.jsonl.
2. Draw --num-pages random ids from that set (seeded).
3. Keep every JSONL record whose *every* gold wikipedia_id lies in the chosen set.
4. Copy matching rows from the source pages DB into a new SQLite file.

Then index with:
  uv run python rag/index.py --db data/wikipedia_pages_hard_10k.sqlite --index-dir data/index_hard_10k --embedded-qdrant
And eval with:
  uv run python eval/run_rag_hard_eval.py --input data/train_hard_subset_10k.jsonl --index-dir data/index_hard_10k ...
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def norm_wid(x: str | int) -> str:
    return str(x).strip()


def collect_ids_and_records(jsonl_path: Path) -> tuple[set[str], list[dict]]:
    mentioned: set[str] = set()
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            for wid in rec.get("wikipedia_id") or []:
                mentioned.add(norm_wid(wid))
    return mentioned, records


def filter_records(records: list[dict], selected: set[str]) -> list[dict]:
    out: list[dict] = []
    for rec in records:
        gold = [norm_wid(w) for w in (rec.get("wikipedia_id") or [])]
        if not gold:
            continue
        if all(w in selected for w in gold):
            out.append(rec)
    return out


def copy_pages_subset(
    src_db: Path,
    dst_db: Path,
    page_ids: set[str],
) -> int:
    if dst_db.exists():
        dst_db.unlink()
    src = sqlite3.connect(src_db)
    dst = sqlite3.connect(dst_db)
    try:
        ddl = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='pages'"
        ).fetchone()
        if not ddl or not ddl[0]:
            raise RuntimeError("Source DB has no 'pages' table")
        dst.execute(ddl[0])
        # copy indexes on pages if any
        for row in src.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='pages' AND sql IS NOT NULL"
        ):
            dst.execute(row[0])

        ids = list(page_ids)
        batch = 400
        inserted = 0
        for i in range(0, len(ids), batch):
            chunk = ids[i : i + batch]
            placeholders = ",".join("?" * len(chunk))
            rows = src.execute(
                f"SELECT wikipedia_id, title, text FROM pages WHERE wikipedia_id IN ({placeholders})",
                chunk,
            ).fetchall()
            dst.executemany(
                "INSERT INTO pages (wikipedia_id, title, text) VALUES (?, ?, ?)",
                rows,
            )
            inserted += len(rows)
        dst.commit()
        return inserted
    finally:
        src.close()
        dst.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset Wikipedia + train_hard by sampled page ids.")
    parser.add_argument("--train-hard", type=Path, default=Path("data/train_hard.jsonl"))
    parser.add_argument("--source-db", type=Path, default=Path("data/wikipedia_pages_50k.sqlite"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/train_hard_subset_10k.jsonl"))
    parser.add_argument("--out-db", type=Path, default=Path("data/wikipedia_pages_hard_10k.sqlite"))
    parser.add_argument("--out-meta", type=Path, default=Path("data/train_hard_subset_10k_meta.json"))
    parser.add_argument("--num-pages", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mentioned, records = collect_ids_and_records(args.train_hard)
    pool = sorted(mentioned)

    src_conn = sqlite3.connect(args.source_db)
    try:
        present = set()
        batch = 500
        for i in range(0, len(pool), batch):
            chunk = pool[i : i + batch]
            ph = ",".join("?" * len(chunk))
            rows = src_conn.execute(
                f"SELECT wikipedia_id FROM pages WHERE wikipedia_id IN ({ph})",
                chunk,
            ).fetchall()
            present.update(norm_wid(r[0]) for r in rows)
    finally:
        src_conn.close()

    pool_in_db = [p for p in pool if p in present]
    n_target = min(args.num_pages, len(pool_in_db))
    if n_target < args.num_pages:
        print(
            f"warning: only {len(pool_in_db)} mentioned ids exist in source DB; using {n_target} pages",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)
    selected = set(rng.sample(pool_in_db, n_target))

    kept = filter_records(records, selected)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as fout:
        for rec in kept:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_copied = copy_pages_subset(args.source_db, args.out_db, selected)

    meta = {
        "seed": args.seed,
        "num_pages_requested": args.num_pages,
        "num_pages_sampled": len(selected),
        "num_pages_copied_to_sqlite": n_copied,
        "ids_mentioned_in_train_hard": len(mentioned),
        "ids_mentioned_and_in_source_db": len(pool_in_db),
        "train_hard_total_records": len(records),
        "train_hard_kept_records": len(kept),
        "train_hard": str(args.train_hard),
        "source_db": str(args.source_db),
        "out_jsonl": str(args.out_jsonl),
        "out_db": str(args.out_db),
    }
    args.out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
