"""Build a hybrid index from structured Wikipedia page JSONL."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


DEFAULT_INPUT = Path("data/structured_pages_50k.jsonl")
DEFAULT_INDEX_DIR = Path("data/index_50k_structured")
EMBED_MODEL = "jinaai/jina-embeddings-v3"
EMBED_BATCH = 128
FLUSH_CHUNKS = 4096
FAISS_INDEX_FILE = "faiss.index"
SECTION_CHUNK_CHARS = 900
FACT_CHUNK_CHARS = 700
SENTENCE_OVERLAP = 1
PAGE_BATCH = 256
MAX_TOC_HEADINGS = 24
MAX_TOC_CHARS = 700

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
PAREN_RE = re.compile(r"\(([^)]+)\)")


def normalize_space(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def first_sentence(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if not text:
        return ""
    sentence = SENTENCE_RE.split(text, maxsplit=1)[0].strip()
    return sentence[:max_chars].strip()


def title_aliases(title: str, infobox: list[dict[str, str]]) -> list[str]:
    aliases = []

    def add(value: str) -> None:
        value = normalize_space(value)
        if value and value not in aliases:
            aliases.append(value)

    add(title)
    plain_title = PAREN_RE.sub("", title).strip()
    add(plain_title)
    for match in PAREN_RE.findall(title):
        add(match)

    alias_keys = {
        "name",
        "fullname",
        "full name",
        "birth_name",
        "birth name",
        "also_known_as",
        "also known as",
        "alias",
        "aliases",
        "nickname",
        "nicknames",
        "title",
    }
    for field in infobox:
        key = str(field.get("key", "")).strip().lower()
        if key in alias_keys:
            add(str(field.get("value", "")))

    return aliases


def table_of_contents(page: dict, max_headings: int = MAX_TOC_HEADINGS, max_chars: int = MAX_TOC_CHARS) -> str:
    headings = []
    for section in page.get("sections") or []:
        heading = normalize_space(str(section.get("heading") or ""))
        if heading:
            headings.append(heading)
        if len(headings) >= max_headings:
            break
    toc = " > ".join(headings)
    if len(toc) > max_chars:
        toc = toc[: max_chars - 3].rstrip() + "..."
    return toc


def chunk_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    chunks = []
    stride = max(1, max_chars - overlap_chars)
    for start in range(0, len(text), stride):
        piece = text[start : start + max_chars].strip()
        if piece:
            chunks.append(piece)
    return chunks


def split_sentences(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def chunk_by_sentences(text: str, max_chars: int, sentence_overlap: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(chunk_long_text(sentence, max_chars=max_chars, overlap_chars=max_chars // 8))
            continue

        next_len = current_len + len(sentence) + (1 if current else 0)
        if current and next_len > max_chars:
            chunks.append(" ".join(current).strip())
            overlap = current[-sentence_overlap:] if sentence_overlap > 0 else []
            current = list(overlap)
            current_len = sum(len(item) for item in current) + max(0, len(current) - 1)

        current.append(sentence)
        current_len += len(sentence) + (1 if current_len else 0)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def chunk_fact(fact: str, max_chars: int) -> list[str]:
    return chunk_long_text(fact, max_chars=max_chars, overlap_chars=max_chars // 10)


def chunk_header(page: dict, chunk_type: str, section: str | None = None) -> str:
    title = str(page.get("title") or "")
    aliases = title_aliases(title, page.get("infobox") or [])
    lead_context = first_sentence(str(page.get("lead") or ""), max_chars=280)
    toc = table_of_contents(page)
    lines = [
        f"Title: {title}",
        f"Aliases: {', '.join(aliases)}",
        f"Chunk type: {chunk_type}",
    ]
    if lead_context:
        lines.append(f"Page context: {lead_context}")
    if toc:
        lines.append(f"Page table of contents: {toc}")
    if section:
        lines.append(f"Section: {section}")
    return "\n".join(lines)


def make_chunk(page: dict, chunk_type: str, body: str, section: str | None = None) -> str:
    return f"{chunk_header(page, chunk_type=chunk_type, section=section)}\n\n{body}".strip()


def page_chunks(page: dict, section_chunk_chars: int, fact_chunk_chars: int, sentence_overlap: int) -> list[str]:
    chunks = []

    lead = str(page.get("lead") or "").strip()
    for piece in chunk_by_sentences(lead, max_chars=section_chunk_chars, sentence_overlap=sentence_overlap):
        chunks.append(make_chunk(page, chunk_type="lead", body=piece, section="Lead"))

    for fact in page.get("facts") or []:
        for piece in chunk_fact(str(fact), max_chars=fact_chunk_chars):
            chunks.append(make_chunk(page, chunk_type="fact", body=piece))

    for section in page.get("sections") or []:
        heading = str(section.get("heading") or "").strip()
        text = str(section.get("text") or "").strip()
        if not heading or not text:
            continue
        for piece in chunk_by_sentences(text, max_chars=section_chunk_chars, sentence_overlap=sentence_overlap):
            chunks.append(make_chunk(page, chunk_type="section", body=piece, section=heading))

    return chunks


def iter_page_batches(path: Path, page_limit: int | None, page_batch: int) -> Iterable[list[dict]]:
    batch = []
    total = 0
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            total += 1
            if page_limit is not None and page_limit > 0 and total >= page_limit:
                yield batch
                return
            if len(batch) >= page_batch:
                yield batch
                batch = []
    if batch:
        yield batch


def count_jsonl(path: Path, page_limit: int | None) -> int:
    total = 0
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                total += 1
                if page_limit is not None and page_limit > 0 and total >= page_limit:
                    return total
    return total


def build_structured_index(
    input_path: Path,
    index_dir: Path,
    section_chunk_chars: int,
    fact_chunk_chars: int,
    sentence_overlap: int,
    page_batch: int,
    embed_batch: int,
    flush_chunks_at: int,
    page_limit: int | None,
    embed_model: str,
) -> None:
    import faiss

    try:
        from rag.embedder_factory import make_embedder
        from rag.index import flush_chunks, init_passages_db, prepare_index_dir
    except ImportError:
        from embedder_factory import make_embedder
        from index import flush_chunks, init_passages_db, prepare_index_dir

    index_dir = prepare_index_dir(index_dir)
    passages_conn = init_passages_db(index_dir / "passages.sqlite")
    faiss_index = None
    embedder: Any | None = None
    vector_dim = None
    chunk_rows = []
    next_chunk_id = 0
    processed_pages = 0
    total_chunks = 0

    total_pages = count_jsonl(input_path, page_limit)

    try:
        embedder = make_embedder(embed_model)
        for pages in tqdm(
            iter_page_batches(input_path, page_limit, page_batch),
            total=(total_pages + page_batch - 1) // page_batch,
            desc="page batches",
        ):
            for page in pages:
                processed_pages += 1
                title = str(page.get("title") or "")
                wikipedia_id = str(page.get("wikipedia_id") or "")
                for chunk_idx, text in enumerate(
                    page_chunks(
                        page,
                        section_chunk_chars=section_chunk_chars,
                        fact_chunk_chars=fact_chunk_chars,
                        sentence_overlap=sentence_overlap,
                    )
                ):
                    chunk_rows.append(
                        {
                            "chunk_id": next_chunk_id,
                            "wikipedia_id": wikipedia_id,
                            "title": title,
                            "chunk_idx": chunk_idx,
                            "text": text,
                        }
                    )
                    next_chunk_id += 1

                if len(chunk_rows) >= flush_chunks_at:
                    faiss_index, vector_dim, uploaded = flush_chunks(
                        chunk_rows=chunk_rows,
                        embedder=embedder,
                        faiss_index=faiss_index,
                        passages_conn=passages_conn,
                        embed_batch=embed_batch,
                        vector_dim=vector_dim,
                    )
                    total_chunks += uploaded
                    chunk_rows.clear()

            if processed_pages % 1000 == 0:
                print(f"processed_pages={processed_pages} | total_chunks={total_chunks}")

        faiss_index, vector_dim, uploaded = flush_chunks(
            chunk_rows=chunk_rows,
            embedder=embedder,
            faiss_index=faiss_index,
            passages_conn=passages_conn,
            embed_batch=embed_batch,
            vector_dim=vector_dim,
        )
        total_chunks += uploaded
        chunk_rows.clear()

        if faiss_index is None or vector_dim is None or total_chunks == 0:
            raise RuntimeError("No structured passages were indexed.")

        faiss.write_index(faiss_index, str(index_dir / FAISS_INDEX_FILE))
        config = {
            "embed_model": embed_model,
            "source": "structured_pages_jsonl",
            "input": str(input_path),
            "schema": "atlas_structured_page_v2",
            "chunking": "section_sentence_and_fact_chunks",
            "section_chunk_chars": section_chunk_chars,
            "fact_chunk_chars": fact_chunk_chars,
            "sentence_overlap": sentence_overlap,
            "max_toc_headings": MAX_TOC_HEADINGS,
            "max_toc_chars": MAX_TOC_CHARS,
            "dim": vector_dim,
            "n_chunks": total_chunks,
            "n_pages": processed_pages,
            "passages_db": "passages.sqlite",
            "dense_backend": "faiss",
            "dense_index_file": FAISS_INDEX_FILE,
        }
        with (index_dir / "config.json").open("w", encoding="utf-8") as fout:
            json.dump(config, fout, ensure_ascii=False, indent=2)

        print({"pages_indexed": processed_pages, "chunks_indexed": total_chunks, "index_dir": str(index_dir)})
    finally:
        passages_conn.close()
        if faiss_index is not None:
            del faiss_index
        if embedder is not None:
            del embedder
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hybrid index from structured Wikipedia page JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--section-chunk-chars", type=int, default=SECTION_CHUNK_CHARS)
    parser.add_argument("--fact-chunk-chars", type=int, default=FACT_CHUNK_CHARS)
    parser.add_argument("--sentence-overlap", type=int, default=SENTENCE_OVERLAP)
    parser.add_argument("--page-batch", type=int, default=PAGE_BATCH)
    parser.add_argument("--embed-batch", type=int, default=EMBED_BATCH)
    parser.add_argument("--flush-chunks", type=int, default=FLUSH_CHUNKS)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    args = parser.parse_args()

    build_structured_index(
        input_path=args.input,
        index_dir=args.index_dir,
        section_chunk_chars=args.section_chunk_chars,
        fact_chunk_chars=args.fact_chunk_chars,
        sentence_overlap=args.sentence_overlap,
        page_batch=args.page_batch,
        embed_batch=args.embed_batch,
        flush_chunks_at=args.flush_chunks,
        page_limit=args.page_limit,
        embed_model=args.embed_model,
    )


if __name__ == "__main__":
    main()
