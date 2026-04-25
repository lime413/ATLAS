from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import sys

sys.path.append(".")

from eval.run_rag_hard_eval import (
    DEFAULT_BERT_MODEL,
    compute_bertscore_max,
    compute_exact_and_f1,
    fetch_server_model_info,
)


def load_existing(results_path: Path) -> tuple[int, float, float, float]:
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


def resume_score(
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
    total, exact_sum, f1_sum, bert_sum = load_existing(results_path)
    print(f"resume_from_scored_examples={total}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
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
        "resumed": True,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume scoring an answers JSONL file without rescoring existing rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-device", default="cuda:0")
    parser.add_argument("--bert-pair-batch-size", type=int, default=384)
    parser.add_argument("--score-chunk-size", type=int, default=128)
    args = parser.parse_args()

    answers_path = args.output_dir / f"{args.run_name}_answers.jsonl"
    results_path = args.output_dir / f"{args.run_name}_results.jsonl"
    summary_path = args.output_dir / f"{args.run_name}_summary.json"

    model_info = fetch_server_model_info(args.base_url)
    resume_score(
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
