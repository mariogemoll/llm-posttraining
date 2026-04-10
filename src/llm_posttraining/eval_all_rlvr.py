# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate all LoRA checkpoints in checkpoints_rlvr with a single vLLM engine load.

Instead of calling `python -m llm_posttraining.eval --backend vllm` once per
checkpoint (which reloads/recompiles the base model each time), this module:

  1. Discovers all checkpoint-* directories in checkpoints_rlvr, sorted by step.
  2. Reads the base model path from the first checkpoint's adapter_config.json.
  3. Initialises one vLLM LLM instance with enable_lora=True.
  4. Loops over every checkpoint, swapping only the LoRARequest – no reload.
  5. Prints a per-checkpoint accuracy table when finished.

Usage (run from the repo root):
    python -m llm_posttraining.eval_all_rlvr
    python -m llm_posttraining.eval_all_rlvr --split test --max_examples 200
    python -m llm_posttraining.eval_all_rlvr --ckpt_dir checkpoints_rlvr --split val
"""

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from llm_posttraining.data import PROMPT_TEMPLATE, load_gsm8k
from llm_posttraining.model import MAX_SEQ_LEN, MODEL_ID
from llm_posttraining.reward import answers_match, extract_answer

# Repo root is three levels up from this file: src/llm_posttraining/eval_all_rlvr.py
_REPO_ROOT = Path(__file__).resolve().parents[2]


def discover_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Return checkpoint-* subdirs sorted by step number."""
    ckpts = sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        raise SystemExit(f"No checkpoint-* directories found in {ckpt_dir}")
    return ckpts


def resolve_base_model(ckpt: Path, fallback: str) -> str:
    """Read base_model_name_or_path from adapter_config.json, resolve relative paths."""
    adapter_cfg_path = ckpt / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise SystemExit(f"{ckpt} does not contain adapter_config.json – not a LoRA checkpoint")
    with open(adapter_cfg_path) as f:
        cfg = json.load(f)
    raw = cfg.get("base_model_name_or_path") or fallback
    # Resolve relative paths against the repo root so vLLM can find them.
    p = Path(raw)
    if not p.is_absolute():
        resolved = (_REPO_ROOT / p).resolve()
        if resolved.exists():
            return str(resolved)
    return raw


def evaluate_checkpoint(
    llm: LLM,
    lora_path: str,
    lora_id: int,
    prompts: list[str],
    gold_answers: list[str],
    max_seq_len: int,
) -> dict:
    """Run generation for one LoRA adapter and return accuracy stats."""
    sampling_params = SamplingParams(temperature=0, max_tokens=max_seq_len)
    lora_request = LoRARequest(f"adapter_{lora_id}", lora_id, lora_path)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    elapsed = time.perf_counter() - t0

    correct = 0
    has_boxed = 0
    total_tokens = 0
    for out, gold in zip(outputs, gold_answers):
        gen = out.outputs[0].text
        total_tokens += len(out.outputs[0].token_ids)
        pred = extract_answer(gen)
        correct += int(answers_match(pred, gold))
        has_boxed += int(bool(re.search(r"\\boxed\{", gen)))

    total = len(gold_answers)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "format_rate": has_boxed / total if total else 0.0,
        "tok_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Eval all RLVR checkpoints with one vLLM load")
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints_rlvr",
        help="Directory containing checkpoint-* subdirs (default: checkpoints_rlvr)",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--model_id", default=MODEL_ID, help="Fallback base model HF ID")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"]
    )
    args = parser.parse_args()

    ckpt_dir = (_REPO_ROOT / args.ckpt_dir).resolve()
    checkpoints = discover_checkpoints(ckpt_dir)
    print(f"Found {len(checkpoints)} checkpoints in {ckpt_dir}:")
    for c in checkpoints:
        print(f"  {c.name}")

    base_model = resolve_base_model(checkpoints[0], args.model_id)
    print(f"\nBase model : {base_model}")
    print(f"Split      : {args.split}")
    print(f"Max seq len: {args.max_seq_len}\n")

    # Load dataset once.
    splits = load_gsm8k()
    dataset = splits[args.split]
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    prompts = [PROMPT_TEMPLATE.format(question=ex["question"]) for ex in dataset]
    gold_answers = [ex["answer"] for ex in dataset]
    print(f"Loaded {len(prompts)} examples from '{args.split}' split.\n")

    # Initialise vLLM once – this is the expensive step.
    print("Initialising vLLM engine (one-time) ...")
    llm = LLM(
        model=base_model,
        dtype=args.dtype,
        enable_lora=True,
        max_model_len=args.max_seq_len,
    )
    print("vLLM engine ready.\n")

    # Evaluate each checkpoint.
    results = []
    for idx, ckpt in enumerate(checkpoints, start=1):
        print(f"[{idx}/{len(checkpoints)}] Evaluating {ckpt.name} ...")
        stats = evaluate_checkpoint(
            llm=llm,
            lora_path=str(ckpt),
            lora_id=idx,
            prompts=prompts,
            gold_answers=gold_answers,
            max_seq_len=args.max_seq_len,
        )
        results.append((ckpt.name, stats))
        print(
            f"  accuracy={stats['accuracy']:.2%}  format={stats['format_rate']:.2%}"
            f"  {stats['tok_per_sec']:.0f} tok/s  ({stats['elapsed']:.1f}s)"
        )

    # Summary table.
    print(f"\n{'=' * 60}")
    print(f"{'Checkpoint':<22} {'Accuracy':>10} {'Format':>10} {'tok/s':>8}")
    print(f"{'-' * 60}")
    for name, s in results:
        print(f"{name:<22} {s['accuracy']:>9.2%} {s['format_rate']:>9.2%} {s['tok_per_sec']:>8.0f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
