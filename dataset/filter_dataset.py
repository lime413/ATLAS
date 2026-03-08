import json
from typing import Any, Dict, List, Optional

import sys
sys.path.append(".")

from eval.metrics import compare_with_gold


def llm_inference(question: str) -> str:
    """
    Placeholder for LLM inference.
    Replace with real model call.
    """
    return "answer from LLM"


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
    batch_size: int = 16,
    device: Optional[str] = None,
    log_every: int = 100,
) -> None:
    """
    Reads train.jsonl, runs placeholder LLM inference on each question,
    evaluates the answer against gold answers, writes:

    1) output_all_path:
       all processed examples with metrics + binary verdict

    2) output_hard_only_path:
       only original train.jsonl records where model did NOT solve the question
    """
    total = 0
    kept_hard = 0
    skipped_no_answers = 0

    fin = open(input_path, "r", encoding="utf-8")
    if results_path is not None:
        fout_all = open(results_path, "w", encoding="utf-8")
    else:
        fout_all = None
    if hard_records_path is not None:
        fout_hard = open(hard_records_path, "w", encoding="utf-8")
    else:
        fout_hard = None

    for line in fin:
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

        llm_answer = llm_inference(question)
        if not isinstance(llm_answer, str):
            llm_answer = str(llm_answer)

        metrics_result = compare_with_gold(
            gold_answers=gold_answers,
            llm_answer=llm_answer,
            batch_size=batch_size,
            device=device,
        )

        verdict = is_model_correct(
            metrics_result=metrics_result
        )

        # Detailed results file
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

        # Save original record in the same format as train.jsonl
        if verdict == 0:
            if fout_hard is not None:
                fout_hard.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_hard += 1

        if total % log_every == 0:
            print(
                f"processed={total} | hard={kept_hard} | skipped_no_answers={skipped_no_answers}"
            )

    print("\nDone")
    print(f"Processed: {total}")
    print(f"Hard examples saved: {kept_hard}")
    print(f"Skipped (no gold answers): {skipped_no_answers}")


if __name__ == "__main__":
    process_train_jsonl(
        input_path="data/train.jsonl",
        results_path=None, # "id", "input", "answer", "llm_answer", "exact_match", "f1","bertscore", "model_solved"
        hard_records_path=None, # hard tasks in same format as train.jsonl
        batch_size=128,
        device="cuda:0"
    )