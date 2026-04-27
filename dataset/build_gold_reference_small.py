from __future__ import annotations

import argparse
import json
import sqlite3
import time
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path("output/index_50k_rawish_50k_all_results.jsonl")
DEFAULT_PAGES_DB = Path("data/wikipedia_pages_50k.sqlite")
DEFAULT_OUTPUT = Path("data/gold_reference_small.jsonl")


BAD_SECTIONS = {
    "references",
    "reference",
    "external links",
    "external link",
    "see also",
    "further reading",
    "notes",
    "footnotes",
    "citations",
    "sources",
    "bibliography",
    "works cited",
    "general references",
    "references and notes",
    "notes and references",
}


COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^/]*/>", re.DOTALL | re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
HEADING_RE = re.compile(r"^\s*(={2,6})\s*(.*?)\s*\1\s*$")
CATEGORY_RE = re.compile(r"\[\[Category:[^\]]+\]\]", re.IGNORECASE)
FILE_RE = re.compile(r"\[\[(?:File|Image):[^\]]+\]\]", re.IGNORECASE)
EXTERNAL_LINK_RE = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
URL_RE = re.compile(r"https?://\S+")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def clean_links(text: str) -> str:
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(1).strip(), text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text


def preprocess_gold_page(
    text: str,
    *,
    keep_infoboxes: bool = False,
    keep_tables: bool = False,
) -> str:
    text = COMMENT_RE.sub("", text or "")
    text = REF_RE.sub("", text)
    text = remove_bad_sections(text)
    text = CATEGORY_RE.sub("", text)
    text = FILE_RE.sub("", text)
    text = remove_table_and_template_blocks(text, keep_infoboxes=keep_infoboxes, keep_tables=keep_tables)
    text = HTML_RE.sub("", text)
    text = clean_links(text)
    text = URL_RE.sub("", text)
    text = re.sub(r"'{2,}", "", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def remove_bad_sections(text: str) -> str:
    out: list[str] = []
    skip_level: int | None = None
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading = re.sub(r"[^A-Za-z0-9 ]+", "", match.group(2)).strip().lower()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if heading in BAD_SECTIONS:
                skip_level = level
                continue
            continue
        if skip_level is not None:
            continue
        out.append(line)
    return "\n".join(out)


def remove_table_and_template_blocks(
    text: str,
    *,
    keep_infoboxes: bool = False,
    keep_tables: bool = False,
) -> str:
    out: list[str] = []
    skip_template = False
    skip_table = False
    depth = 0
    for line in text.splitlines():
        stripped = line.strip()
        if skip_table:
            if stripped.startswith("|}"):
                skip_table = False
            continue
        if stripped.startswith("{|") and not keep_tables:
            skip_table = True
            continue
        if skip_template:
            depth += stripped.count("{{") - stripped.count("}}")
            if depth <= 0:
                skip_template = False
            continue
        if stripped.startswith("{{") and not (keep_infoboxes and stripped.lower().startswith("{{infobox")):
            depth = stripped.count("{{") - stripped.count("}}")
            if depth > 0:
                skip_template = True
            continue
        out.append(line)
    return "\n".join(out)


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    previous_space = True
    for idx, char in enumerate(text.casefold()):
        if char.isalnum():
            chars.append(char)
            index_map.append(idx)
            previous_space = False
        elif not previous_space:
            chars.append(" ")
            index_map.append(idx)
            previous_space = True
    if chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()
    return "".join(chars), index_map


def find_answer_span(text: str, answers: list[str]) -> tuple[int | None, int | None, str | None]:
    normalized_text, index_map = normalize_with_map(text)
    for answer in sorted((a for a in answers if a), key=len, reverse=True):
        normalized_answer = NON_WORD_RE.sub(" ", answer.casefold()).strip()
        if not normalized_answer:
            continue
        pos = normalized_text.find(normalized_answer)
        if pos >= 0:
            start = index_map[pos]
            end = index_map[pos + len(normalized_answer) - 1] + 1
            return start, end, answer
    return None, None, None


def answer_window(text: str, answers: list[str], window_chars: int) -> tuple[str, int, int, str | None, bool]:
    start, end, matched = find_answer_span(text, answers)
    if start is None or end is None:
        chunk = text[:window_chars]
        return chunk, 0, len(chunk), None, False

    center = (start + end) // 2
    chunk_start = max(0, center - window_chars // 2)
    chunk_end = min(len(text), chunk_start + window_chars)
    chunk_start = max(0, chunk_end - window_chars)
    return text[chunk_start:chunk_end], chunk_start, chunk_end, matched, True


def choose_gold_chunk(text: str, answers: list[str], window_chars: int) -> dict[str, object]:
    modes = [
        ("without_infoboxes_or_tables", False, False),
        ("with_infoboxes", True, False),
        ("with_tables", False, True),
        ("with_infoboxes_and_tables", True, True),
    ]

    fallback_text = ""
    fallback_mode = "with_infoboxes_and_tables"
    for mode, keep_infoboxes, keep_tables in modes:
        preprocessed = preprocess_gold_page(text, keep_infoboxes=keep_infoboxes, keep_tables=keep_tables)
        chunk, start, end, matched, found = answer_window(preprocessed, answers, window_chars)
        if mode == fallback_mode:
            fallback_text = preprocessed
        if found:
            return {
                "preprocessing_mode": mode,
                "preprocessed_chars": len(preprocessed),
                "chunk": chunk,
                "chunk_start": start,
                "chunk_end": end,
                "matched_answer": matched,
                "found_answer": True,
            }

    chunk = fallback_text[:window_chars]
    return {
        "preprocessing_mode": fallback_mode,
        "preprocessed_chars": len(fallback_text),
        "chunk": chunk,
        "chunk_start": 0,
        "chunk_end": len(chunk),
        "matched_answer": None,
        "found_answer": False,
    }


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
