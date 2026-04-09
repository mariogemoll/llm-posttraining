"""Verifiable reward for GSM8K: exact-match on numeric answer in \\boxed{}."""

import re


def extract_answer(text: str) -> str:
    """Extract the number from \\boxed{} in generated text."""
    m = re.search(r"\\boxed\{([\-\d,\.]+)\}", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return ""


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted and gold answers match numerically."""
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred.strip() == gold.strip()


def compute_reward(generated: str, gold_answer: str) -> float:
    """Return 1.0 if the generated text contains the correct answer in \\boxed{}, else 0.0."""
    pred = extract_answer(generated)
    return 1.0 if answers_match(pred, gold_answer) else 0.0
