from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    from dataset.inspect_gold_pages_preprocessing import choose_gold_chunk
except ImportError:
    from inspect_gold_pages_preprocessing import choose_gold_chunk


DEFAULT_RESULTS = Path("output/index_50k_rawish_50k_all_results.jsonl")
DEFAULT_PAGES_DB = Path("data/wikipedia_pages_50k.sqlite")
DEFAULT_OUTPUT = Path("data/gold_reference_small.jsonl")


class PageCache:
    def __init__(self, conn: sqlite3.Connection, max_pages: int) -> None:
        self.conn = conn
        self.max_pages = max(0, max_pages)
        self.items: OrderedDict[str, dict[str, Any] | None] = OrderedDict()

    def get(self, wikipedia_id: str) -> dict[str, Any] | None:
        wikipedia_id = str(wikipedia_id)
        if self.max_pages > 0 and wikipedia_id in self.items:
            value = self.items.pop(wikipedia_id)
            self.items[wikipedia_id] = value
            return value

        row = self.conn.execute(
            "SELECT wikipedia_id, title, text FROM pages WHERE wikipedia_id = ?",
            (wikipedia_id,),
        ).fetchone()
        value = None
        if row is not None:
            raw_text = row["text"] or ""
            value = {
                "wikipedia_id": str(row["wikipedia_id"]),
                "title": row["title"] or "",
                "raw_text": raw_text,
                "raw_chars": len(raw_text),
            }

        if self.max_pages > 0:
            self.items[wikipedia_id] = value
            while len(self.items) > self.max_pages:
                self.items.popitem(last=False)
        return value


def iter_records(path: Path):
    with path.open(encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def build_gold_reference_small(
    results_path: Path,
    pages_db: Path,
    output_path: Path,
    chunk_chars: int,
    limit: int | None,
    cache_pages: int,
    log_every: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(pages_db)
    conn.row_factory = sqlite3.Row
    cache = PageCache(conn, max_pages=cache_pages)

    records_seen = 0
    rows_written = 0
    pages_missing = 0
    answer_found = 0
    answer_not_found = 0
    started_at = time.time()

    try:
        with output_path.open("w", encoding="utf-8") as fout:
            for line_no, record in iter_records(results_path):
                if limit is not None and records_seen >= limit:
                    break
                records_seen += 1

                question_id = str(record.get("id", ""))
                question = str(record.get("input", ""))
                gold_answers = [str(answer) for answer in (record.get("gold_answers") or [])]
                gold_ids = [str(wid) for wid in (record.get("wikipedia_id") or [])]

                for page_rank, wikipedia_id in enumerate(gold_ids, 1):
                    page = cache.get(wikipedia_id)
                    if page is None:
                        pages_missing += 1
                        continue

                    selected = choose_gold_chunk(
                        page["raw_text"],
                        gold_answers,
                        chunk_chars,
                    )
                    found = bool(selected["found_answer"])
                    if not found:
                        answer_not_found += 1
                        continue
                    if found:
                        answer_found += 1

                    row = {
                        "id": question_id,
                        "input_line": line_no,
                        "question": question,
                        "gold_answers": gold_answers,
                        "gold_page_rank": page_rank,
                        "wikipedia_id": page["wikipedia_id"],
                        "title": page["title"],
                        "matched_answer": selected["matched_answer"],
                        "found_answer": found,
                        "preprocessing_mode": selected["preprocessing_mode"],
                        "chunk_start": selected["chunk_start"],
                        "chunk_end": selected["chunk_end"],
                        "raw_chars": page["raw_chars"],
                        "preprocessed_chars": selected["preprocessed_chars"],
                        "chunk_chars": len(str(selected["chunk"])),
                        "gold_reference_chunk": selected["chunk"],
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1

                if log_every > 0 and records_seen % log_every == 0:
                    elapsed = time.time() - started_at
                    print(
                        f"records={records_seen} rows={rows_written} "
                        f"answer_found={answer_found} answer_not_found={answer_not_found} "
                        f"missing_pages={pages_missing} "
                        f"elapsed_min={elapsed / 60:.1f}"
                    )

        summary = {
            "records_seen": records_seen,
            "rows_written": rows_written,
            "pages_missing": pages_missing,
            "answer_found": answer_found,
            "answer_not_found_skipped": answer_not_found,
            "answer_found_rate": answer_found / (answer_found + answer_not_found) if answer_found + answer_not_found else 0.0,
            "results_path": str(results_path),
            "pages_db": str(pages_db),
            "output_path": str(output_path),
            "chunk_chars": chunk_chars,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small gold reference chunks for RAG mistake analysis.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pages-db", type=Path, default=DEFAULT_PAGES_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-chars", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of result records to process.")
    parser.add_argument("--cache-pages", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=5000)
    args = parser.parse_args()

    build_gold_reference_small(
        results_path=args.results,
        pages_db=args.pages_db,
        output_path=args.output,
        chunk_chars=args.chunk_chars,
        limit=args.limit,
        cache_pages=args.cache_pages,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
