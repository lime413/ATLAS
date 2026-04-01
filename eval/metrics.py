import re
import string
from collections import Counter
from typing import List, Dict, Any

try:
    from bert_score import score as bertscore
except ImportError:
    bertscore = None


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """
    Normalize answer string:
    1) lowercase
    2) remove punctuation
    3) remove articles: a, an, the
    4) collapse multiple spaces
    """
    if text is None:
        return ""

    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def exact_match_score(prediction: str, gold: str) -> float:
    """
    Exact Match after normalization.
    """
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1_score(prediction: str, gold: str) -> float:
    """
    Token-level F1 after normalization.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def bertscore_pairwise(
    prediction: str,
    gold_answers: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    batch_size: int = 128,
    device: str = None,
    rescale_with_baseline: bool = True,
) -> List[float]:
    """
    Compute BERTScore F1 for each (prediction, gold_i) pair.
    Returns a list of scores of the same length as gold_answers.
    On raw text, not normalized text.
    """
    if not gold_answers:
        return []

    if bertscore is None:
        raise ImportError("bert_score is not installed, but use_bertscore=True was requested.")

    predictions = [prediction] * len(gold_answers)
    _, _, f1 = bertscore(
        cands=predictions,
        refs=gold_answers,
        model_type=model_type,
        lang=lang,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )
    return f1.tolist()


def compare_with_gold(
    gold_answers: List[str],
    llm_answer: str,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    batch_size: int = 16,
    device: str = None,
    rescale_with_baseline: bool = True,
    use_bertscore: bool = True,
) -> Dict[str, Any]:
    """
    For each gold answer:
      - normalize texts
      - compute Exact Match
      - compute token-level F1
      - compute BERTScore
    """
    if not isinstance(gold_answers, list):
        raise TypeError("gold_answers must be a list of strings")
    if not all(isinstance(x, str) for x in gold_answers):
        raise TypeError("Every element in gold_answers must be a string")
    if not isinstance(llm_answer, str):
        raise TypeError("llm_answer must be a string")

    if len(gold_answers) == 0:
        return {
            "best_exact_match": 0.0,
            "best_f1": 0.0,
            "best_bertscore": 0.0
        }

    if use_bertscore:
        bertscores = bertscore_pairwise(
            prediction=llm_answer,
            gold_answers=gold_answers,
            model_type=model_type,
            lang=lang,
            batch_size=batch_size,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
        )
    else:
        bertscores = [0.0] * len(gold_answers)

    all_pairs = []
    for gold, bs in zip(gold_answers, bertscores):
        pair_result = {
            "gold_answer": gold,
            "llm_answer": llm_answer,
            "normalized_gold": normalize_answer(gold),
            "normalized_llm": normalize_answer(llm_answer),
            "exact_match": exact_match_score(llm_answer, gold),
            "token_f1": token_f1_score(llm_answer, gold),
            "bertscore_f1": float(bs),
        }
        all_pairs.append(pair_result)

    best_exact_match_pair = max(all_pairs, key=lambda x: x["exact_match"])
    best_f1_pair = max(all_pairs, key=lambda x: x["token_f1"])
    best_bertscore_pair = max(all_pairs, key=lambda x: x["bertscore_f1"])

    return {
        "best_exact_match": best_exact_match_pair["exact_match"],
        "best_f1": best_f1_pair["token_f1"],
        "best_bertscore": best_bertscore_pair["bertscore_f1"]
    }


if __name__ == "__main__":
    gold_answers = [
        "the therefore sign",
        "therefore sign",
        "a logical consequence, such as the conclusion of a syllogism",
        "the therefore sign ( ∴ ) is generally used before a logical consequence , such as the conclusion of a syllogism",
    ]

    llm_answer = "The three dots mean the therefore sign."

    result = compare_with_gold(
        gold_answers=gold_answers,
        llm_answer=llm_answer,
        batch_size=128,
        device="cuda:0",
    )