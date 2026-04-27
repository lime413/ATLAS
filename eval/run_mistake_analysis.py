from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

DEFAULT_INPUT = Path("output/index_50k_rawish_50k_all_results.jsonl")
DEFAULT_OUTPUT = Path("output/index_50k_rawish_50k_all_answer_scores.json")
DEFAULT_TEMPLATE = Path("eval/rag_answer_score_prompt_template.md")
DEFAULT_GOLD_REFERENCE_SMALL = Path("data/gold_reference_small.jsonl")
DEFAULT_PASSAGES_DB = Path("data/index_50k_rawish/passages.sqlite")
DEFAULT_BASE_URL = "http://127.0.0.1:8082/v1"
DEFAULT_MODEL = "gemma-3-4b-it-Q4_K_M.gguf"
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TOKEN_F1_CORRECT_THRESHOLD = 0.8
DEFAULT_BERTSCORE_CORRECT_THRESHOLD = 0.9


def load_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```(?:text)?\n(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def load_existing(path: Path, overwrite: bool) -> dict[str, Any]:
    if overwrite or not path.exists():
        return {"items": []}
    with path.open(encoding="utf-8") as fin:
        payload = json.load(fin)
    if isinstance(payload, list):
        payload = {"items": payload}
    elif not (isinstance(payload, dict) and isinstance(payload.get("items"), list)):
        raise ValueError(f"Unsupported output JSON shape: {path}")

    for item in payload["items"]:
        if "output" not in item and "analysis" in item:
            item["output"] = item.pop("analysis")
        elif "analysis" in item:
            item.pop("analysis")
    return payload


def save_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def is_likely_correct(record: dict[str, Any], token_f1_threshold: float, bertscore_threshold: float) -> bool:
    exact_match = float(record.get("exact_match") or 0.0)
    token_f1 = float(record.get("token_f1") or 0.0)
    bertscore_f1 = float(record.get("bertscore_f1") or 0.0)
    return exact_match >= 1.0 or token_f1 > token_f1_threshold or bertscore_f1 > bertscore_threshold


def iter_records(
    path: Path,
    include_correct: bool,
    seen_ids: set[str],
    token_f1_threshold: float,
    bertscore_threshold: float,
):
    with path.open(encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", ""))
            if sample_id in seen_ids:
                continue
            if not include_correct and is_likely_correct(record, token_f1_threshold, bertscore_threshold):
                continue
            yield line_no, record


def open_passages_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_gold_reference_chunks(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record.get("id", ""))
            wid = str(record.get("wikipedia_id", ""))
            if not qid or not wid:
                continue
            by_question.setdefault(qid, {})[wid] = record
    return by_question


def fetch_passages(conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
    ids = [int(chunk_id) for chunk_id in chunk_ids]
    if not ids:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
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
            chunk_id = int(row["chunk_id"])
            out[chunk_id] = {
                "chunk_id": chunk_id,
                "wikipedia_id": str(row["wikipedia_id"]),
                "title": row["title"] or "",
                "chunk_idx": int(row["chunk_idx"]),
                "text": row["text"] or "",
            }
    return out


def gold_refs_for_record(
    record: dict[str, Any],
    gold_chunks: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    question_id = str(record.get("id", ""))
    by_page = gold_chunks.get(question_id, {})
    refs: list[dict[str, Any]] = []
    for wid in [str(item) for item in (record.get("wikipedia_id") or [])]:
        ref = by_page.get(wid)
        if ref is None:
            return None
        refs.append(ref)
    return refs


def format_gold_reference_chunks(refs: list[dict[str, Any]]) -> str:
    parts = []
    for rank, ref in enumerate(refs, 1):
        parts.append(
            f"{rank}. Page id: {ref.get('wikipedia_id', '')}\n"
            f"   Title: {ref.get('title', '')}\n"
            f"   Gold page rank: {ref.get('gold_page_rank', rank)}\n"
            f"   Preprocessing mode: {ref.get('preprocessing_mode', '')}\n"
            f"   Matched answer: {ref.get('matched_answer', '')}\n"
            f"   Chunk window: {ref.get('chunk_start', '')}-{ref.get('chunk_end', '')}\n"
            f"   Text:\n"
            f"   {ref.get('gold_reference_chunk', '')}"
        )
    return "\n\n".join(parts)


def format_retrieved_chunks(
    sources: list[dict[str, Any]],
    passages: dict[int, dict[str, Any]],
) -> str:
    parts = []
    for rank, source in enumerate(sources, 1):
        chunk_id = int(source.get("chunk_id"))
        passage = passages.get(chunk_id)
        if passage is None:
            parts.append(
                f"{rank}. Chunk id: {chunk_id}\n"
                f"   Page id: {source.get('wikipedia_id', '')}\n"
                f"   Title: {source.get('title', '')}\n"
                f"   Rank: {rank}\n"
                f"   Text:\n"
                f"   [chunk text not found in passages.sqlite]"
            )
            continue
        parts.append(
            f"{rank}. Chunk id: {chunk_id}\n"
            f"   Page id: {passage.get('wikipedia_id', '')}\n"
            f"   Title: {passage.get('title', '')}\n"
            f"   Rank: {rank}\n"
            f"   Chunk index: {passage.get('chunk_idx', '')}\n"
            f"   Text:\n"
            f"   {passage.get('text', '')}"
        )
    return "\n\n".join(parts) if parts else "No chunks were retrieved."


def has_all_gold_references(
    record: dict[str, Any],
    gold_chunks: dict[str, dict[str, dict[str, Any]]],
) -> bool:
    return gold_refs_for_record(record, gold_chunks) is not None


def selected_sources(record: dict[str, Any], max_chunks: int) -> list[dict[str, Any]]:
    sources = []
    for source in record.get("sources") or []:
        if source.get("chunk_id") is None:
            continue
        sources.append(source)
        if len(sources) >= max_chunks:
            break
    return sources


def build_prompt(
    template: str,
    record: dict[str, Any],
    gold_chunks: dict[str, dict[str, dict[str, Any]]],
    passages_conn: sqlite3.Connection,
    max_retrieved_chunks: int,
) -> str | None:
    gold_refs = gold_refs_for_record(record, gold_chunks)
    if gold_refs is None:
        return None

    sources = selected_sources(record, max_retrieved_chunks)
    passages = fetch_passages(passages_conn, [int(source["chunk_id"]) for source in sources])

    replacements = {
        "{QUESTION}": str(record.get("input", "")),
        "{GOLD_ANSWERS}": json.dumps(record.get("gold_answers") or [], ensure_ascii=False),
        "{GOLD_REFERENCE_PAGES}": format_gold_reference_chunks(gold_refs),
        "{RETRIEVED_PAGES}": format_retrieved_chunks(sources, passages),
        "{RETRIEVED_CHUNKS}": format_retrieved_chunks(sources, passages),
        "{SYSTEM_ANSWER}": str(record.get("llm_answer", "")),
        "{EXACT_MATCH}": str(record.get("exact_match", "")),
        "{TOKEN_F1}": str(record.get("token_f1", "")),
        "{BERTSCORE_F1}": str(record.get("bertscore_f1", "")),
    }

    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    text = choices[0].get("text")
    return text.strip() if isinstance(text, str) else ""


def call_model(base_url: str, model: str, prompt: str, max_output_tokens: int, retries: int) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    request_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url,
                json=request_payload,
                timeout=120,
            )
            response.raise_for_status()
            return extract_chat_completion_text(response.json())
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Model request failed: {last_error}") from last_error


def wait_for_rate_limit(requests_sent: int, requests_per_minute: int, pbar: tqdm | None) -> None:
    if requests_per_minute <= 0 or requests_sent <= 0 or requests_sent % requests_per_minute != 0:
        return
    if pbar is not None:
        pbar.set_postfix(wait_s=60, refresh=True)
    time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a local LLM server to judge RAG answers.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="RAG results JSONL file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON file.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Prompt template markdown file.")
    parser.add_argument("--gold-reference-small", default=str(DEFAULT_GOLD_REFERENCE_SMALL))
    parser.add_argument("--passages-db", default=str(DEFAULT_PASSAGES_DB))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Local LLM server /v1 URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by the local judging server.")
    parser.add_argument("--limit", type=int, default=0, help="Number of new samples to process. Use 0 for no limit.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Max tokens in the model response.")
    parser.add_argument("--max-retrieved-chunks", type=int, default=5)
    parser.add_argument(
        "--include-correct",
        action="store_true",
        default=False,
        help="Process likely correct rows. This is disabled by default.",
    )
    parser.add_argument("--token-f1-correct-threshold", type=float, default=DEFAULT_TOKEN_F1_CORRECT_THRESHOLD)
    parser.add_argument("--bertscore-correct-threshold", type=float, default=DEFAULT_BERTSCORE_CORRECT_THRESHOLD)
    parser.add_argument("--include-prompts", action="store_true", help="Store filled prompts in the output JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output instead of resuming.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and output JSON without API calls.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-sample log lines and show only tqdm progress.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    sample_limit = args.limit if args.limit > 0 else None

    template = load_template(Path(args.template))
    payload = load_existing(output_path, args.overwrite)
    seen_ids = {str(item.get("id")) for item in payload.get("items", [])}
    gold_chunks = load_gold_reference_chunks(Path(args.gold_reference_small))

    payload["meta"] = {
        "input": str(input_path),
        "template": str(args.template),
        "gold_reference_small": str(args.gold_reference_small),
        "passages_db": str(args.passages_db),
        "base_url": args.base_url,
        "model": args.model,
        "limit": sample_limit,
        "output_mode": args.output_mode,
        "max_retrieved_chunks": args.max_retrieved_chunks,
        "include_correct": bool(args.include_correct),
        "skip_correct_rule": {
            "exact_match": ">= 1.0",
            "token_f1": f"> {args.token_f1_correct_threshold}",
            "bertscore_f1": f"> {args.bertscore_correct_threshold}",
        },
    }
    payload.setdefault("items", [])

    skipped_missing_gold = 0
    processed = 0
    scanned = 0
    requests_sent = 0
    with open_passages_db(Path(args.passages_db)) as passages_conn, tqdm(
        total=sample_limit,
        desc="processing",
        unit="sample",
    ) as pbar:
        for line_no, record in iter_records(
            path=input_path,
            include_correct=args.include_correct,
            seen_ids=seen_ids,
            token_f1_threshold=args.token_f1_correct_threshold,
            bertscore_threshold=args.bertscore_correct_threshold,
        ):
            if sample_limit is not None and processed >= sample_limit:
                break
            scanned += 1
            prompt = build_prompt(
                template=template,
                record=record,
                gold_chunks=gold_chunks,
                passages_conn=passages_conn,
                max_retrieved_chunks=args.max_retrieved_chunks,
            )
            if prompt is None:
                skipped_missing_gold += 1
                pbar.set_postfix(scanned=scanned, skipped_missing_gold=skipped_missing_gold, refresh=False)
                continue
            output = (
                "DRY RUN: prompt was built but not sent."
                if args.dry_run
                else call_model(args.base_url, args.model, prompt, args.max_output_tokens, args.retries).strip()
            )
            if not args.dry_run:
                requests_sent += 1
            processed += 1
            item = {
                "id": record.get("id"),
                "output": output,
                "input_line": line_no,
                "exact_match": record.get("exact_match"),
                "token_f1": record.get("token_f1"),
                "bertscore_f1": record.get("bertscore_f1"),
                "skipped_missing_gold_before": skipped_missing_gold,
            }
            if args.include_prompts:
                item["prompt"] = prompt
            payload["items"].append(item)
            payload["meta"]["skipped_missing_gold"] = skipped_missing_gold
            payload["meta"]["scanned_candidates"] = scanned
            payload["meta"]["requests_sent"] = requests_sent
            payload["meta"]["requests_per_minute"] = args.requests_per_minute
            save_output(output_path, payload)
            pbar.update(1)
            pbar.set_postfix(scanned=scanned, skipped_missing_gold=skipped_missing_gold, refresh=False)
            if not args.quiet:
                tqdm.write(f"processed={processed} id={record.get('id')}")
            if sample_limit is None or processed < sample_limit:
                wait_for_rate_limit(requests_sent, args.requests_per_minute, pbar)


if __name__ == "__main__":
    main()
