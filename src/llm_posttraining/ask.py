#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One-turn chat with a trained model.

Usage:
    python -m llm_posttraining.ask "Janet has 5 apples..."
    python -m llm_posttraining.ask --checkpoint checkpoints_rlvr/checkpoint-1200 "Janet has 5 apples..."
    python -m llm_posttraining.ask --checkpoint checkpoints_sft/merged "Janet has 5 apples..."
    python -m llm_posttraining.ask --max-new-tokens 512 "Janet has 5 apples..."
"""

import argparse
import json
import sys
from pathlib import Path
from threading import Thread

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

from llm_posttraining.data import PROMPT_TEMPLATE
from llm_posttraining.model import MODEL_ID

_REPO_ROOT = Path(__file__).parent.parent.parent

BOLD_YELLOW = "\033[1;33m"
GREY = "\033[90m"
RESET = "\033[0m"
_BOXED_OPEN = r"\boxed{"


def _load_tokenizer(path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(path)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(checkpoint: Path | None) -> tuple:
    """Load model and tokenizer.

    - No checkpoint: base Qwen model from HuggingFace.
    - Checkpoint with adapter_config.json: LoRA adapter; base model is read
      from the adapter config's base_model_name_or_path.
    - Checkpoint without adapter_config.json: plain fine-tuned model.
    """
    if checkpoint is None:
        model_path = MODEL_ID
        print(f"Loading base model {model_path} ...", file=sys.stderr)
        tokenizer = _load_tokenizer(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tokenizer

    adapter_config = checkpoint / "adapter_config.json"
    if adapter_config.exists():
        base_path = json.loads(adapter_config.read_text())["base_model_name_or_path"]
        # Resolve relative paths against the repo root
        if not Path(base_path).is_absolute():
            base_path = str(_REPO_ROOT / base_path)
        print(f"Loading base model from {base_path} ...", file=sys.stderr)
        tokenizer = _load_tokenizer(str(checkpoint))
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        print(f"Loading LoRA adapter from {checkpoint} ...", file=sys.stderr)
        model = PeftModel.from_pretrained(base, str(checkpoint))
    else:
        print(f"Loading model from {checkpoint} ...", file=sys.stderr)
        tokenizer = _load_tokenizer(str(checkpoint))
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tokenizer


def chat_streaming(question: str, model, tokenizer, max_new_tokens: int = 512) -> None:
    """Stream the response, printing tokens as they arrive.

    Buffers just enough text to detect \\boxed{...} across chunk boundaries
    and prints the answer in bold yellow instead.
    """
    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        ),
    )
    thread.start()

    pending = ""
    inside_boxed = False

    for chunk in streamer:
        pending += chunk
        while True:
            if not inside_boxed:
                idx = pending.find(_BOXED_OPEN)
                if idx == -1:
                    # Keep a small tail buffered in case \boxed{ is split across chunks
                    flush_up_to = max(0, len(pending) - len(_BOXED_OPEN))
                    print(pending[:flush_up_to], end="", flush=True)
                    pending = pending[flush_up_to:]
                    break
                else:
                    print(pending[:idx], end="", flush=True)
                    pending = pending[idx + len(_BOXED_OPEN) :]
                    inside_boxed = True
            else:
                idx = pending.find("}")
                if idx == -1:
                    break
                print(f"{BOLD_YELLOW}{pending[:idx]}{RESET}", end="", flush=True)
                pending = pending[idx + 1 :]
                inside_boxed = False

    print(pending, flush=True)
    thread.join()


def main():
    parser = argparse.ArgumentParser(description="One-turn chat with a trained model.")
    parser.add_argument("question", nargs="+", help="Math question to ask the model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory (plain model or LoRA adapter). Omit for base Qwen.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    prompt = PROMPT_TEMPLATE.format(question=question)
    print(f"\n{GREY}{prompt}{RESET}", flush=True)
    chat_streaming(question, model, tokenizer, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
