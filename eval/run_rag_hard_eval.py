import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
import torch
from bert_score import score as bertscore
from openai import OpenAI

sys.path.append(".")

from eval.metrics import exact_match_score, token_f1_score
from rag.constants import DEFAULT_LLM_MODEL
from rag.search import close_index, load_index, retrieve

DEFAULT_INPUT = Path("data/train_hard.jsonl")
DEFAULT_INDEX_DIR = Path("data/index_50k_rawish")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = DEFAULT_LLM_MODEL
DEFAULT_BERT_MODEL = "microsoft/deberta-xlarge-mnli"


def apply_rag_memory_env(
    *,
    low_mem: bool,
    embed_torch_dtype: str | None,
    torch_intraop_threads: int | None,
    torch_interop_threads: int | None,
    sqlite_cache_kb: int | None,
    sqlite_mmap_mb: int | None,
) -> None:
    """Tune env vars read by rag.embedder_factory and rag.search.load_index before loading models."""
    if low_mem:
        os.environ.setdefault("ATLAS_TORCH_INTRAOP_THREADS", "2")
        os.environ.setdefault("ATLAS_TORCH_INTEROP_THREADS", "1")
        os.environ.setdefault("ATLAS_EMBED_TORCH_DTYPE", "float16")
        os.environ.setdefault("ATLAS_SQLITE_CACHE_KB", "2048")
        os.environ.setdefault("ATLAS_SQLITE_MMAP_MB", "64")
    if embed_torch_dtype:
        os.environ["ATLAS_EMBED_TORCH_DTYPE"] = embed_torch_dtype.strip().lower()
    if torch_intraop_threads is not None:
        os.environ["ATLAS_TORCH_INTRAOP_THREADS"] = str(torch_intraop_threads)
    if torch_interop_threads is not None:
        os.environ["ATLAS_TORCH_INTEROP_THREADS"] = str(torch_interop_threads)
    if sqlite_cache_kb is not None:
        os.environ["ATLAS_SQLITE_CACHE_KB"] = str(sqlite_cache_kb)
    if sqlite_mmap_mb is not None:
        os.environ["ATLAS_SQLITE_MMAP_MB"] = str(sqlite_mmap_mb)


def fetch_server_model_info(base_url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {"base_url": base_url, "error": str(exc)}

    model_id = None
    if payload.get("data"):
        model_id = payload["data"][0].get("id")
    elif payload.get("models"):
        model_id = payload["models"][0].get("model")

    return {
        "base_url": base_url,
        "model_id": model_id,
        "raw": payload,
    }


def build_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    context = "\n\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    return (
        "Answer the question using ONLY the provided context. Be concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answers(
    input_path: Path,
    output_path: Path,
    index_dir: Path,
    base_url: str,
    model: str,
    top_k: int,
    dense_limit: int,
    sparse_limit: int,
    sparse_weight: float,
    max_tokens: int,
    log_every: int,
    query_embed_batch: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = load_index(index_dir)
    client = OpenAI(base_url=base_url, api_key="not-needed")

    beb = max(1, int(query_embed_batch))
    processed = 0
    started_at = time.time()
    pending: list[dict[str, Any]] = []

    def flush_pending(fout) -> None:
        nonlocal pending, processed
        if not pending:
            return
        batch = pending
        pending = []
        questions = [r.get("input", "") for r in batch]
        with torch.inference_mode():
            mat = index["embedder"].encode(
                questions,
                task="retrieval.query",
                normalize_embeddings=True,
                batch_size=min(beb, len(questions)),
            )
        for record, qemb in zip(batch, mat):
            question = record.get("input", "")
            gold_answers = record.get("answer", [])
            passages = retrieve(
                query=question,
                index=index,
                top_k=top_k,
                dense_limit=dense_limit,
                sparse_limit=sparse_limit,
                sparse_weight=sparse_weight,
                query_embedding=qemb,
            )
            prompt = build_prompt(question, passages)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            llm_answer = (response.choices[0].message.content or "").strip()

            answer_record = {
                "id": record.get("id"),
                "input": question,
                "wikipedia_id": record.get("wikipedia_id", []),
                "gold_answers": gold_answers,
                "llm_answer": llm_answer,
                "sources": [
                    {
                        "chunk_id": passage["chunk_id"],
                        "wikipedia_id": passage["wikipedia_id"],
                        "title": passage["title"],
                        "chunk_idx": passage["chunk_idx"],
                    }
                    for passage in passages
                ],
            }
            fout.write(json.dumps(answer_record, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            if processed % log_every == 0:
                elapsed = time.time() - started_at
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(f"generated={processed} | elapsed_min={elapsed/60:.1f} | qps={rate:.3f}")

    try:
        with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                pending.append(json.loads(line))
                if len(pending) >= beb:
                    flush_pending(fout)
            flush_pending(fout)
    finally:
        close_index(index)


def compute_exact_and_f1(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    exact_scores: list[float] = []
    f1_scores: list[float] = []

    for record in records:
        prediction = record["llm_answer"]
        gold_answers = record["gold_answers"]
        if not gold_answers:
            exact_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        em = max(exact_match_score(prediction, gold) for gold in gold_answers)
        f1 = max(token_f1_score(prediction, gold) for gold in gold_answers)
        exact_scores.append(float(em))
        f1_scores.append(float(f1))

    return exact_scores, f1_scores


def compute_bertscore_max(
    records: list[dict[str, Any]],
    model_type: str,
    batch_size: int,
    device: str,
    pair_batch_size: int,
) -> list[float]:
    candidates: list[str] = []
    references: list[str] = []
    pair_example_idx: list[int] = []

    for example_idx, record in enumerate(records):
        prediction = record["llm_answer"]
        for gold in record["gold_answers"]:
            candidates.append(prediction)
            references.append(gold)
            pair_example_idx.append(example_idx)

    best_scores = [0.0] * len(records)
    if not candidates:
        return best_scores

    for start in range(0, len(candidates), pair_batch_size):
        end = min(start + pair_batch_size, len(candidates))
        _, _, f1 = bertscore(
            cands=candidates[start:end],
            refs=references[start:end],
            model_type=model_type,
            lang="en",
            batch_size=batch_size,
            device=device,
            rescale_with_baseline=True,
            verbose=False,
        )
        for local_idx, score in enumerate(f1.tolist()):
            example_idx = pair_example_idx[start + local_idx]
            if score > best_scores[example_idx]:
                best_scores[example_idx] = float(score)

        processed_pairs = end
        print(f"bertscore_pairs={processed_pairs}/{len(candidates)}")

    return best_scores


def evaluate_answers(
    answers_path: Path,
    results_path: Path,
    summary_path: Path,
    model_info: dict[str, Any],
    bert_model: str,
    bert_batch_size: int,
    bert_device: str,
    bert_pair_batch_size: int,
    score_chunk_size: int,
) -> None:
    total = 0
    exact_sum = 0.0
    f1_sum = 0.0
    bert_sum = 0.0

    def iter_chunks():
        chunk: list[dict[str, Any]] = []
        with answers_path.open(encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                chunk.append(json.loads(line))
                if len(chunk) >= score_chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

    with results_path.open("w", encoding="utf-8") as fout:
        for chunk_idx, records in enumerate(iter_chunks(), 1):
            exact_scores, f1_scores = compute_exact_and_f1(records)
            bert_scores = compute_bertscore_max(
                records=records,
                model_type=bert_model,
                batch_size=bert_batch_size,
                device=bert_device,
                pair_batch_size=bert_pair_batch_size,
            )

            for record, exact_score, f1_score, bert_score_value in zip(records, exact_scores, f1_scores, bert_scores):
                result_record = {
                    **record,
                    "exact_match": exact_score,
                    "token_f1": f1_score,
                    "bertscore_f1": bert_score_value,
                }
                fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            total += len(records)
            exact_sum += sum(exact_scores)
            f1_sum += sum(f1_scores)
            bert_sum += sum(bert_scores)
            fout.flush()
            print(f"scored_chunks={chunk_idx} | scored_examples={total}")

            del records, exact_scores, f1_scores, bert_scores
            gc.collect()

    summary = {
        "total_examples": total,
        "mean_exact_match": exact_sum / total if total else 0.0,
        "mean_token_f1": f1_sum / total if total else 0.0,
        "mean_bertscore_f1": bert_sum / total if total else 0.0,
        "llm_server": model_info,
        "bertscore_model": bert_model,
        "answers_path": str(answers_path),
        "results_path": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full RAG evaluation on hard questions and save answers plus metrics.")
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "score"],
        default="all",
        help=(
            "Execution stage. "
            "'generate' runs retrieval + LLM answers only, "
            "'score' computes EM/F1/BERTScore from saved answers only, "
            "'all' runs generation first and then launches scoring in a fresh Python process to keep RAM usage lower."
        ),
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default=None, help="Optional prefix for output files. Defaults to index dir name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-limit", type=int, default=20)
    parser.add_argument("--sparse-limit", type=int, default=20)
    parser.add_argument("--sparse-weight", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--query-embed-batch",
        type=int,
        default=32,
        help="Batch size for query embedding during generate (larger = fewer forward passes; watch VRAM). Use 1 to mimic old per-question encode.",
    )
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help=(
            "Enable a conservative preset: float16 embedder weights, fewer PyTorch threads, smaller SQLite cache/mmap. "
            "If embeddings look wrong, drop this flag or set --embed-torch-dtype float32."
        ),
    )
    parser.add_argument(
        "--embed-torch-dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Override embedder weight dtype (also set ATLAS_EMBED_TORCH_DTYPE). float16 roughly halves transformer RAM.",
    )
    parser.add_argument("--torch-intraop-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    parser.add_argument(
        "--sqlite-cache-kb",
        type=int,
        default=None,
        help="SQLite page cache size in KiB (PRAGMA cache_size, negative value semantics).",
    )
    parser.add_argument(
        "--sqlite-mmap-mb",
        type=int,
        default=None,
        help="SQLite mmap cap in MB (0 disables mmap for passages DB).",
    )
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-device", default="cuda:0")
    parser.add_argument("--bert-pair-batch-size", type=int, default=384)
    parser.add_argument(
        "--score-chunk-size",
        type=int,
        default=128,
        help="How many answer records to load into memory at once during scoring.",
    )
    args = parser.parse_args()

    apply_rag_memory_env(
        low_mem=args.low_mem,
        embed_torch_dtype=args.embed_torch_dtype,
        torch_intraop_threads=args.torch_intraop_threads,
        torch_interop_threads=args.torch_interop_threads,
        sqlite_cache_kb=args.sqlite_cache_kb,
        sqlite_mmap_mb=args.sqlite_mmap_mb,
    )

    input_path = Path(args.input)
    index_dir = Path(args.index_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or index_dir.name
    answers_path = output_dir / f"{run_name}_answers.jsonl"
    results_path = output_dir / f"{run_name}_results.jsonl"
    summary_path = output_dir / f"{run_name}_summary.json"
    model_info_path = output_dir / f"{run_name}_model_info.json"

    model_info = fetch_server_model_info(args.base_url)
    model_info_path.write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")


    if args.stage in {"all", "generate"}:
        generate_answers(
            input_path=input_path,
            output_path=answers_path,
            index_dir=index_dir,
            base_url=args.base_url,
            model=args.model,
            top_k=args.top_k,
            dense_limit=args.dense_limit,
            sparse_limit=args.sparse_limit,
            sparse_weight=args.sparse_weight,
            max_tokens=args.max_tokens,
            log_every=args.log_every,
            query_embed_batch=args.query_embed_batch,
        )

    if args.stage == "all":
        gc.collect()
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__)),
                "--stage",
                "score",
                "--input",
                str(input_path),
                "--index-dir",
                str(index_dir),
                "--output-dir",
                str(output_dir),
                "--run-name",
                run_name,
                "--base-url",
                args.base_url,
                "--model",
                args.model,
                "--bert-model",
                args.bert_model,
                "--bert-batch-size",
                str(args.bert_batch_size),
                "--bert-device",
                args.bert_device,
                "--bert-pair-batch-size",
                str(args.bert_pair_batch_size),
                "--score-chunk-size",
                str(args.score_chunk_size),
            ],
            check=True,
        )
        return

    if args.stage == "score":
        evaluate_answers(
            answers_path=answers_path,
            results_path=results_path,
            summary_path=summary_path,
            model_info=model_info,
            bert_model=args.bert_model,
            bert_batch_size=args.bert_batch_size,
            bert_device=args.bert_device,
            bert_pair_batch_size=args.bert_pair_batch_size,
            score_chunk_size=args.score_chunk_size,
        )


if __name__ == "__main__":
    main()
