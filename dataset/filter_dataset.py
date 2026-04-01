import argparse
import json
from typing import Any, Dict, List, Optional

import sys
sys.path.append(".")

from openai import OpenAI

from eval.metrics import compare_with_gold


DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "local-model"


def llm_inference(
    question: str,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 64,
) -> str:
    """
    Run local OpenAI-compatible inference without retrieval/context.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer factual questions only from your own knowledge. "
                    "Do not mention sources, documents, or Wikipedia. "
                    "If you do not know the answer, reply exactly with: I don't know."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Answer briefly and directly.\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content
    return (content or "").strip()


def extract_gold_answers(record: Dict[str, Any]) -> List[str]:
    """
    Extract gold answers from processed train.jsonl record:
    {
        "id": "...",
        "input": "...",
        "answer": ["...", "..."]
    }
    """
    answers = record.get("answer", [])

    if not isinstance(answers, list):
        return []

    cleaned = []
    for ans in answers:
        if isinstance(ans, str):
            ans = ans.strip()
            if ans:
                cleaned.append(ans)

    # deduplicate while preserving order
    seen = set()
    unique_answers = []
    for ans in cleaned:
        if ans not in seen:
            seen.add(ans)
            unique_answers.append(ans)

    return unique_answers


def is_model_correct(
    metrics_result: Dict[str, float],
    f1_threshold: float = 0.8,
    bertscore_threshold: float = 0.9,
) -> int:
    """
    Returns binary verdict:
      1 -> model solved the question
      0 -> model did not solve it

    Solved if ANY of:
      - Exact Match == 1
      - token F1 >= f1_threshold
      - BERTScore >= bertscore_threshold
    """
    em = float(metrics_result.get("best_exact_match", 0.0))
    f1 = float(metrics_result.get("best_f1", 0.0))
    bs = float(metrics_result.get("best_bertscore", 0.0))

    return int((em == 1.0) or (f1 >= f1_threshold) or (bs >= bertscore_threshold))


def process_train_jsonl(
    input_path: str,
    results_path: Optional[str] = None,
    hard_records_path: Optional[str] = None,
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL,
    max_tokens: int = 64,
    batch_size: int = 16,
    device: Optional[str] = None,
    use_bertscore: bool = False,
    log_every: int = 100,
    limit: Optional[int] = None,
) -> None:
    """
    Reads train.jsonl, runs LLM inference on each question without retrieval,
    evaluates the answer against gold answers, writes:

    1) output_all_path:
       all processed examples with metrics + binary verdict

    2) output_hard_only_path:
       only original train.jsonl records where model did NOT solve the question
    """
    total = 0
    kept_hard = 0
    skipped_no_answers = 0

    client = OpenAI(base_url=llm_base_url, api_key="not-needed")

    with open(input_path, "r", encoding="utf-8") as fin:
        if results_path is not None:
            fout_all = open(results_path, "w", encoding="utf-8")
        else:
            fout_all = None
        if hard_records_path is not None:
            fout_hard = open(hard_records_path, "w", encoding="utf-8")
        else:
            fout_hard = None

        try:
            for line in fin:
                if limit is not None and total >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                total += 1
                record = json.loads(line)

                question = record.get("input", "")
                if not isinstance(question, str):
                    question = str(question)

                gold_answers = extract_gold_answers(record)
                if not gold_answers:
                    skipped_no_answers += 1
                    continue

                llm_answer = llm_inference(
                    question=question,
                    client=client,
                    model=llm_model,
                    max_tokens=max_tokens,
                )
                if not isinstance(llm_answer, str):
                    llm_answer = str(llm_answer)

                metrics_result = compare_with_gold(
                    gold_answers=gold_answers,
                    llm_answer=llm_answer,
                    batch_size=batch_size,
                    device=device,
                    use_bertscore=use_bertscore,
                )

                verdict = is_model_correct(
                    metrics_result=metrics_result
                )

                result_record = {
                    "id": record.get("id"),
                    "input": question,
                    "answer": gold_answers,
                    "llm_answer": llm_answer,
                    "exact_match": metrics_result["best_exact_match"],
                    "f1": metrics_result["best_f1"],
                    "bertscore": metrics_result["best_bertscore"],
                    "model_solved": verdict,
                }
                if fout_all is not None:
                    fout_all.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    fout_all.flush()

                if verdict == 0:
                    if fout_hard is not None:
                        fout_hard.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout_hard.flush()
                    kept_hard += 1

                if total % log_every == 0:
                    print(
                        f"processed={total} | hard={kept_hard} | skipped_no_answers={skipped_no_answers}"
                    )
        finally:
            if fout_all is not None:
                fout_all.close()
            if fout_hard is not None:
                fout_hard.close()

    print("\nDone")
    print(f"Processed: {total}")
    print(f"Hard examples saved: {kept_hard}")
    print(f"Skipped (no gold answers): {skipped_no_answers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset by questions the LLM cannot answer without retrieval.")
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument("--results", default="data/filter_results.jsonl")
    parser.add_argument("--hard-records", default="data/train_hard.jsonl")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-bertscore", action="store_true")
    args = parser.parse_args()

    process_train_jsonl(
        input_path=args.input,
        results_path=args.results,
        hard_records_path=args.hard_records,
        llm_base_url=args.base_url,
        llm_model=args.model,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        device=args.device,
        use_bertscore=args.use_bertscore,
        log_every=args.log_every,
        limit=args.limit,
    )
