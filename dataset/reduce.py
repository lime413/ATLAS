import argparse
import json
import random
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Iterable

DATA_DIR = Path("data")
DEFAULT_TRAIN_PATH = DATA_DIR / "train.jsonl"
DEFAULT_SOURCE_DB = DATA_DIR / "wikipedia_pages.sqlite"
DEFAULT_OUTPUT_DB = DATA_DIR / "wikipedia_pages_50k.sqlite"
DEFAULT_TARGET_PAGES = 50_000
DEFAULT_SEED = 42
SQLITE_VAR_LIMIT = 500


def get_relevant_ids(train_path: Path) -> set[str]:
    ids: set[str] = set()
    with train_path.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            for wikipedia_id in record.get("wikipedia_id", []):
                ids.add(str(wikipedia_id))
    return ids


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def init_output_db(output_db: Path) -> sqlite3.Connection:
    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(output_db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE pages (
            wikipedia_id TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            ns           INTEGER,
            text         TEXT
        )
        """
    )
    conn.commit()
    return conn


def copy_rows_by_ids(
    source_conn: sqlite3.Connection,
    output_conn: sqlite3.Connection,
    wikipedia_ids: set[str],
    log_every: int = 2_000,
) -> int:
    source_cur = source_conn.cursor()
    output_cur = output_conn.cursor()
    copied = 0

    items = list(wikipedia_ids)
    for batch in batched(items, SQLITE_VAR_LIMIT):
        placeholders = ",".join("?" * len(batch))
        rows = source_cur.execute(
            f"""
            SELECT wikipedia_id, title, ns, text
            FROM pages
            WHERE wikipedia_id IN ({placeholders})
            """,
            batch,
        ).fetchall()
        output_cur.executemany(
            "INSERT OR IGNORE INTO pages (wikipedia_id, title, ns, text) VALUES (?, ?, ?, ?)",
            rows,
        )
        copied += len(rows)
        if copied % log_every == 0:
            print(f"copied relevant pages: {copied}")

    output_conn.commit()
    return copied


def sample_random_non_relevant_rows(
    source_conn: sqlite3.Connection,
    excluded_ids: set[str],
    sample_size: int,
    seed: int,
    log_every: int = 1_000,
) -> list[tuple[str, str, int, str]]:
    if sample_size <= 0:
        return []

    cur = source_conn.cursor()
    min_rowid, max_rowid = cur.execute("SELECT MIN(rowid), MAX(rowid) FROM pages").fetchone()
    if min_rowid is None or max_rowid is None:
        return []

    rng = random.Random(seed)
    selected: dict[str, tuple[str, str, int, str]] = {}
    attempts = 0

    while len(selected) < sample_size:
        rowid = rng.randint(min_rowid, max_rowid)
        attempts += 1
        row = cur.execute(
            """
            SELECT wikipedia_id, title, ns, text
            FROM pages
            WHERE rowid = ?
            """,
            (rowid,),
        ).fetchone()
        if row is None:
            continue

        wikipedia_id = str(row[0])
        if wikipedia_id in excluded_ids or wikipedia_id in selected:
            continue

        selected[wikipedia_id] = row
        if len(selected) % log_every == 0:
            print(
                f"sampled random pages: {len(selected)} / {sample_size} | attempts={attempts}"
            )

    print(f"random sampling finished | selected={len(selected)} | attempts={attempts}")
    return list(selected.values())


def create_indexes(output_conn: sqlite3.Connection) -> None:
    output_conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_title ON pages(title)")
    output_conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_ns ON pages(ns)")
    output_conn.commit()


def reduce_dataset(
    train_path: Path,
    source_db: Path,
    output_db: Path,
    target_pages: int,
    seed: int,
) -> None:
    started_at = time.time()

    relevant_ids = get_relevant_ids(train_path)
    print(f"relevant wikipedia ids in train: {len(relevant_ids):,}")
    if len(relevant_ids) > target_pages:
        raise ValueError(
            f"target_pages={target_pages} is smaller than the number of relevant ids={len(relevant_ids)}"
        )

    source_conn = sqlite3.connect(source_db)
    output_conn = init_output_db(output_db)

    try:
        copied_relevant = copy_rows_by_ids(source_conn, output_conn, relevant_ids)
        print(f"copied relevant pages to output: {copied_relevant:,}")

        remaining = target_pages - copied_relevant
        print(f"need additional random pages: {remaining:,}")

        random_rows = sample_random_non_relevant_rows(
            source_conn=source_conn,
            excluded_ids=relevant_ids,
            sample_size=remaining,
            seed=seed,
        )

        if len(random_rows) != remaining:
            raise RuntimeError(
                f"Expected {remaining} random rows, but sampled {len(random_rows)}"
            )

        output_conn.executemany(
            "INSERT OR IGNORE INTO pages (wikipedia_id, title, ns, text) VALUES (?, ?, ?, ?)",
            random_rows,
        )
        output_conn.commit()

        create_indexes(output_conn)

        total_rows = output_conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        print(f"final output rows: {total_rows:,}")
        if total_rows != target_pages:
            raise RuntimeError(
                f"Expected {target_pages} rows in output db, but found {total_rows}"
            )
    finally:
        source_conn.close()
        output_conn.close()

    elapsed = time.time() - started_at
    size_gb = output_db.stat().st_size / (1024 ** 3)
    print(
        {
            "output_db": str(output_db),
            "target_pages": target_pages,
            "relevant_pages": copied_relevant,
            "random_pages": remaining,
            "size_gb": round(size_gb, 3),
            "elapsed_min": round(elapsed / 60, 2),
            "seed": seed,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reduced Wikipedia sqlite with all train-relevant pages plus random filler pages."
    )
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB))
    parser.add_argument("--output-db", default=str(DEFAULT_OUTPUT_DB))
    parser.add_argument("--target-pages", type=int, default=DEFAULT_TARGET_PAGES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    reduce_dataset(
        train_path=Path(args.train),
        source_db=Path(args.source_db),
        output_db=Path(args.output_db),
        target_pages=args.target_pages,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
