# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""GSM8K dataset loading and preprocessing."""

import re

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "Problem: {question}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Solution: "
)


def extract_gold_answer(answer_text: str) -> str:
    """Extract the numeric answer after '####' in a GSM8K answer field."""
    match = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def format_chain(chain: str, answer: str) -> str:
    """Clean a GSM8K chain-of-thought for use as an SFT target.

    - Strips inline calculator annotations like <<3*4=12>>
    - Replaces '#### N' with '\\boxed{N}' to match the reward format
    """
    chain = re.sub(r"<<[^>]+>>", "", chain)
    chain = re.sub(r"####\s*[\d,.\-]+", f"\\\\boxed{{{answer}}}", chain)
    return chain.strip()


def load_gsm8k(val_size: int = 500, seed: int = 42) -> dict:
    """Load GSM8K with train/val/test splits.

    The val set is carved from train (deterministic with seed=42).

    Returns dict with keys 'train', 'val', 'test'. Each split is a
    HuggingFace Dataset with columns: question, chain, answer.
    """
    raw = load_dataset("openai/gsm8k", "main")

    def process(example):
        return {
            "question": example["question"].strip(),
            "chain": example["answer"].strip(),
            "answer": extract_gold_answer(example["answer"]),
        }

    train_all = raw["train"].map(process, remove_columns=["question", "answer"])
    test = raw["test"].map(process, remove_columns=["question", "answer"])

    split = train_all.train_test_split(test_size=val_size, seed=seed)

    return {"train": split["train"], "val": split["test"], "test": test}
