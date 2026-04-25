from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

sys.path.append(".")

from eval.run_rag_hard_eval import (
    DEFAULT_BERT_MODEL,
    compute_bertscore_max,
    compute_exact_and_f1,
    fetch_server_model_info,
)
from rag.constants import DEFAULT_LLM_MODEL


DEFAULT_INPUT = Path("data/train_hard.jsonl")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = DEFAULT_LLM_MODEL


def build_prompt(question: str) -> str:
    return f"Answer the question directly and briefly.\n\nQuestion: {question}\nAnswer:"


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open(encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Broken JSON in {path} at line {line_no}. Fix or remove this line before resume.") from exc
            total += 1
    return total


def generate_answers(
    input_path: Path,
    output_path: Path,
    base_url: str,
    model: str,
    max_tokens: int,
    log_every: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=base_url, api_key="not-needed")
    resume_from = count_jsonl_records(output_path)
    processed = resume_from
    generated_now = 0
    started_at = time.time()
    print(f"resume_from_generated_answers={resume_from}")

    input_seen = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            if input_seen < resume_from:
                input_seen += 1
                continue
            input_seen += 1
            record = json.loads(line)
            question = record.get("input", "")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_prompt(question)}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            llm_answer = (response.choices[0].message.content or "").strip()
            answer_record: dict[str, Any] = {
                "id": record.get("id"),
                "input": question,
                "wikipedia_id": record.get("wikipedia_id", []),
                "gold_answers": record.get("answer", []),
                "llm_answer": llm_answer,
                "sources": [],
            }
            fout.write(json.dumps(answer_record, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            generated_now += 1
            if processed % log_every == 0:
                elapsed = time.time() - started_at
                rate = generated_now / elapsed if elapsed > 0 else 0.0
                print(f"generated={processed} | elapsed_min={elapsed/60:.1f} | qps={rate:.3f}")

    print(f"done_generation | generated_total={processed} | generated_now={generated_now}")


def load_existing_results(results_path: Path) -> tuple[int, float, float, float]:
    if not results_path.exists():
        return 0, 0.0, 0.0, 0.0

    total = 0
    exact_sum = 0.0
    f1_sum = 0.0
    bert_sum = 0.0
    with results_path.open(encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Broken JSON in {results_path} at line {line_no}. Fix or remove this line before resume.") from exc
            total += 1
            exact_sum += float(record.get("exact_match", 0.0))
            f1_sum += float(record.get("token_f1", 0.0))
            bert_sum += float(record.get("bertscore_f1", 0.0))
    return total, exact_sum, f1_sum, bert_sum


def iter_answer_chunks(answers_path: Path, skip: int, chunk_size: int):
    chunk: list[dict[str, Any]] = []
    seen = 0
    with answers_path.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            if seen < skip:
                seen += 1
                continue
            seen += 1
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def score_answers(
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
    answer_total = count_jsonl_records(answers_path)
    total, exact_sum, f1_sum, bert_sum = load_existing_results(results_path)
    if total > answer_total:
        raise ValueError(f"Results contain {total} rows, but answers contain only {answer_total} rows.")
    print(f"resume_from_scored_examples={total}")

    with results_path.open("a", encoding="utf-8") as fout:
        for chunk_idx, records in enumerate(iter_answer_chunks(answers_path, total, score_chunk_size), 1):
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
        "answer_total_at_scoring": answer_total,
        "resumable": True,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-only evaluation without retrieval.")
    parser.add_argument("--stage", choices=["all", "generate", "score"], default="all")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="llm_only_train_hard")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-device", default="cuda:0")
    parser.add_argument("--bert-pair-batch-size", type=int, default=384)
    parser.add_argument("--score-chunk-size", type=int, default=128)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    answers_path = output_dir / f"{args.run_name}_answers.jsonl"
    results_path = output_dir / f"{args.run_name}_results.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.json"
    model_info_path = output_dir / f"{args.run_name}_model_info.json"

    model_info = fetch_server_model_info(args.base_url)
    model_info_path.write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.stage in {"all", "generate"}:
        generate_answers(
            input_path=input_path,
            output_path=answers_path,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            log_every=args.log_every,
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
                "--output-dir",
                str(output_dir),
                "--run-name",
                args.run_name,
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
        score_answers(
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
