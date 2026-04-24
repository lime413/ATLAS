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

from eval.run_rag_hard_eval import DEFAULT_BERT_MODEL, evaluate_answers, fetch_server_model_info
from rag.constants import DEFAULT_LLM_MODEL


DEFAULT_INPUT = Path("data/train_hard.jsonl")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = DEFAULT_LLM_MODEL


def build_prompt(question: str) -> str:
    return f"Answer the question directly and briefly.\n\nQuestion: {question}\nAnswer:"


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
    processed = 0
    started_at = time.time()

    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
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
            if processed % log_every == 0:
                elapsed = time.time() - started_at
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(f"generated={processed} | elapsed_min={elapsed/60:.1f} | qps={rate:.3f}")


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
