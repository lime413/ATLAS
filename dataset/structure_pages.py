from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable


HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^/]*/>", re.DOTALL | re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image):[^\]]+\]\]", re.IGNORECASE)
CATEGORY_RE = re.compile(r"\[\[Category:[^\]]+\]\]", re.IGNORECASE)
EXTERNAL_LINK_RE = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
WIKI_LINK_WITH_LABEL_RE = re.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")
WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
TEMPLATE_NAME_RE = re.compile(r"^\s*\{\{\s*([^|\n{}]+)", re.IGNORECASE)


def normalize_space(text: str) -> str:
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_balanced_blocks(text: str, start_token: str, end_token: str) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    pos = 0
    while True:
        start = text.find(start_token, pos)
        if start == -1:
            break
        depth = 1
        i = start + len(start_token)
        while i < len(text):
            if text.startswith(start_token, i):
                depth += 1
                i += len(start_token)
            elif text.startswith(end_token, i):
                depth -= 1
                i += len(end_token)
                if depth == 0:
                    blocks.append((start, i, text[start:i]))
                    break
            else:
                i += 1
        if depth != 0:
            break
        pos = i
    return blocks


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts = []
    prev = 0
    for start, end in sorted(spans):
        parts.append(text[prev:start])
        prev = end
    parts.append(text[prev:])
    return "".join(parts)


def split_top_level(text: str, sep: str) -> list[str]:
    parts = []
    start = 0
    brace_depth = 0
    link_depth = 0
    i = 0
    while i < len(text):
        if text.startswith("{{", i):
            brace_depth += 1
            i += 2
            continue
        if text.startswith("}}", i) and brace_depth > 0:
            brace_depth -= 1
            i += 2
            continue
        if text.startswith("[[", i):
            link_depth += 1
            i += 2
            continue
        if text.startswith("]]", i) and link_depth > 0:
            link_depth -= 1
            i += 2
            continue
        if text.startswith(sep, i) and brace_depth == 0 and link_depth == 0:
            parts.append(text[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return parts


def clean_value(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = REF_RE.sub("", text)
    text = FILE_LINK_RE.sub("", text)
    text = CATEGORY_RE.sub("", text)
    text = EXTERNAL_LINK_RE.sub(r"\1", text)
    text = WIKI_LINK_WITH_LABEL_RE.sub(r"\1", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\{\{convert\|([^|{}]+)\|([^|{}]+).*?\}\}", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{birth date(?: and age)?\|([^|{}]+)\|([^|{}]+)\|([^|{}]+).*?\}\}", r"\1-\2-\3", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = HTML_TAG_RE.sub("", text)
    return normalize_space(text)


def is_infobox(block: str) -> bool:
    match = TEMPLATE_NAME_RE.match(block)
    return bool(match and match.group(1).strip().lower().startswith("infobox"))


def parse_infobox(block: str) -> list[dict[str, str]]:
    inner = block.strip()
    if inner.startswith("{{"):
        inner = inner[2:]
    if inner.endswith("}}"):
        inner = inner[:-2]

    fields = []
    for part in split_top_level(inner, "\n|"):
        if "=" not in part:
            continue
        raw_key, raw_value = part.split("=", 1)
        key = clean_value(raw_key.strip().lstrip("|"))
        value = clean_value(raw_value)
        if key and value:
            fields.append({"key": key, "value": value})
    return fields


def parse_table(block: str, max_rows: int) -> dict[str, object] | None:
    caption = ""
    headers: list[str] = []
    rows: list[list[str]] = []
    current: list[str] = []

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("{|") or line.startswith("|}"):
            continue
        if line.startswith("|+"):
            caption = clean_value(line[2:])
            continue
        if line.startswith("|-"):
            if current:
                rows.append(current)
                current = []
            continue
        if line.startswith("!"):
            headers.extend(clean_value(cell) for cell in split_top_level(line[1:], "!!") if clean_value(cell))
            continue
        if line.startswith("|"):
            cells = [clean_value(cell) for cell in split_top_level(line[1:], "||")]
            cells = [cell for cell in cells if cell]
            if cells:
                current.extend(cells)

    if current:
        rows.append(current)

    rows = rows[:max_rows]
    if not caption and not headers and not rows:
        return None
    return {
        "caption": caption,
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
    }


def extract_infoboxes(text: str) -> tuple[list[dict[str, str]], list[tuple[int, int]]]:
    infoboxes: list[dict[str, str]] = []
    spans = []
    for start, end, block in find_balanced_blocks(text, "{{", "}}"):
        if not is_infobox(block):
            continue
        infoboxes.extend(parse_infobox(block))
        spans.append((start, end))
    return infoboxes, spans


def extract_tables(text: str, max_rows: int) -> tuple[list[dict[str, object]], list[tuple[int, int]]]:
    tables = []
    spans = []
    for start, end, block in find_balanced_blocks(text, "{|", "|}"):
        table = parse_table(block, max_rows=max_rows)
        if table is not None:
            tables.append(table)
            spans.append((start, end))
    return tables, spans


def parse_sections(text: str) -> tuple[str, list[dict[str, object]]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return clean_value(text), []

    lead = clean_value(text[: matches[0].start()])
    sections = []
    for idx, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = clean_value(match.group(2))
        body = clean_value(text[content_start:content_end])
        if heading and body:
            sections.append(
                {
                    "heading": heading,
                    "level": len(match.group(1)),
                    "text": body,
                }
            )
    return lead, sections


def structure_page(wikipedia_id: str, title: str, raw_text: str, max_table_rows: int) -> dict[str, object]:
    text = COMMENT_RE.sub("", raw_text or "")
    infobox, infobox_spans = extract_infoboxes(text)
    tables, table_spans = extract_tables(text, max_rows=max_table_rows)
    text_without_blocks = remove_spans(text, infobox_spans + table_spans)
    lead, sections = parse_sections(text_without_blocks)
    return {
        "schema_version": "atlas_structured_page_v1",
        "wikipedia_id": str(wikipedia_id),
        "title": title,
        "lead": lead,
        "infobox": infobox,
        "sections": sections,
        "tables": tables,
        "stats": {
            "infobox_fields": len(infobox),
            "sections": len(sections),
            "tables": len(tables),
            "raw_chars": len(raw_text or ""),
            "lead_chars": len(lead),
        },
    }


def iter_pages(db_path: Path, limit: int | None) -> Iterable[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    query = "SELECT wikipedia_id, title, text FROM pages ORDER BY wikipedia_id"
    params: tuple[int, ...] = ()
    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params = (limit,)
    try:
        for row in conn.execute(query, params):
            yield row
    finally:
        conn.close()


def write_structured_jsonl(db_path: Path, output: Path, limit: int | None, max_table_rows: int, log_every: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as fout:
        for row in iter_pages(db_path, limit):
            record = structure_page(
                wikipedia_id=row["wikipedia_id"],
                title=row["title"],
                raw_text=row["text"] or "",
                max_table_rows=max_table_rows,
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            if total % log_every == 0:
                print(f"structured_pages={total}")
    print(f"done | structured_pages={total} | output={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw Wikipedia pages from SQLite to structured JSONL.")
    parser.add_argument("--db", type=Path, default=Path("data/wikipedia_pages_50k.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("data/structured_pages_sample.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-table-rows", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    write_structured_jsonl(
        db_path=args.db,
        output=args.output,
        limit=args.limit,
        max_table_rows=args.max_table_rows,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
