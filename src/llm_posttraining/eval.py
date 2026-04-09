# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate a model on GSM8K using HuggingFace generation.

Works with base models, merged checkpoints, and LoRA adapter checkpoints.
Supports CUDA, MPS (macOS), and CPU.

Usage:
    python -m llm_posttraining.eval --split val
    python -m llm_posttraining.eval --split val --ckpt checkpoints_sft/merged
    python -m llm_posttraining.eval --split val --ckpt checkpoints_trl/checkpoint-200
    python -m llm_posttraining.eval --split test --show_samples 10
"""

import argparse
import json
import os
import random
import re
import time

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_posttraining.data import PROMPT_TEMPLATE, load_gsm8k
from llm_posttraining.reward import answers_match, extract_answer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_MAX_SEQ_LEN = 384


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str):
    if requested != "auto":
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[requested]
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(ckpt: str | None, device: str, dtype, model_id: str = DEFAULT_MODEL_ID):
    """Load model and tokenizer, auto-detecting LoRA vs merged checkpoints."""
    is_lora = bool(ckpt and os.path.exists(os.path.join(ckpt, "adapter_config.json")))

    if is_lora:
        with open(os.path.join(ckpt, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)
        model_path = adapter_cfg.get("base_model_name_or_path") or model_id
    else:
        model_path = ckpt or model_id

    label = f"LoRA adapter from {ckpt}" if is_lora else (f"merged model from {ckpt}" if ckpt else f"base model {model_id}")
    print(f"Loading {label} (device={device}, dtype={dtype}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    if is_lora:
        model = PeftModel.from_pretrained(model, ckpt, is_trainable=False)

    model.to(device)
    model.eval()
    return model, tokenizer


def generate_batched(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int):
    """Generate completions in batches. Yields (start_idx, texts, token_counts)."""
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_new_tokens)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_lengths = inputs["attention_mask"].sum(dim=1)

        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        texts, token_counts = [], []
        for seq, plen in zip(sequences, prompt_lengths):
            gen_ids = seq[int(plen) :]
            texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
            token_counts.append(int(gen_ids.numel()))

        yield start, texts, token_counts


def evaluate(
    split: str = "val",
    max_examples: int | None = None,
    ckpt: str | None = None,
    batch_size: int = 4,
    device: str = "auto",
    dtype: str = "auto",
    model_id: str = DEFAULT_MODEL_ID,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
):
    splits = load_gsm8k()
    dataset = splits[split]
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    device = detect_device(device)
    dtype = pick_dtype(device, dtype)
    model, tokenizer = load_model_and_tokenizer(ckpt=ckpt, device=device, dtype=dtype, model_id=model_id)

    prompts = [PROMPT_TEMPLATE.format(question=ex["question"]) for ex in dataset]
    gold_answers = [ex["answer"] for ex in dataset]

    print(f"Evaluating {len(prompts)} examples ({split} split) ...")
    wall_start = time.perf_counter()

    correct = 0
    has_boxed = 0
    total_tokens = 0
    results = []

    for start, generations, token_counts in generate_batched(model, tokenizer, prompts, batch_size, max_seq_len):
        total_tokens += sum(token_counts)
        batch_gold = gold_answers[start : start + len(generations)]

        for i, (gen, gold) in enumerate(zip(generations, batch_gold)):
            pred = extract_answer(gen)
            ok = answers_match(pred, gold)
            correct += int(ok)
            has_boxed += int(bool(re.search(r"\\boxed\{", gen)))
            results.append({
                "question": dataset[start + i]["question"],
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "generated": gen,
            })

    elapsed = time.perf_counter() - wall_start
    total = len(results)
    acc = correct / total if total else 0.0
    format_rate = has_boxed / total if total else 0.0
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    print(f"\n{'=' * 55}")
    print(f"Split       : {split}")
    print(f"Examples    : {total}")
    print(f"Correct     : {correct}")
    print(f"Accuracy    : {acc:.2%}")
    print(f"Format rate : {format_rate:.2%}  (\\boxed{{}} present)")
    print(f"Elapsed     : {elapsed:.1f}s  ({tok_per_sec:.0f} tok/s)")
    print(f"{'=' * 55}")

    return acc, results


def show_samples(results: list[dict], n: int):
    sample = random.sample(results, min(n, len(results)))
    for i, r in enumerate(sample):
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"\n--- Sample {i + 1} [{status}] ---")
        print(f"Q: {r['question'][:200]}")
        print(f"Generated:\n{r['generated']}")
        print(f"Predicted: {r['pred']}  |  Gold: {r['gold']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--ckpt", default=None, help="Merged model dir or LoRA adapter dir")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Base model HF ID")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--show_samples", type=int, default=0, help="Print N random completions")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    args = parser.parse_args()

    acc, results = evaluate(
        split=args.split,
        max_examples=args.max_examples,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        model_id=args.model_id,
        max_seq_len=args.max_seq_len,
    )
    if args.show_samples:
        show_samples(results, args.show_samples)


if __name__ == "__main__":
    main()
