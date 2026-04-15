#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One-turn chat with a trained model.

Usage:
    python -m llm_posttraining.ask "Janet has 5 apples..."
    python -m llm_posttraining.ask --model checkpoints_rlvr/checkpoint-1200 "Janet has 5 apples..."
    python -m llm_posttraining.ask --model checkpoints_sft/merged "Janet has 5 apples..."
    python -m llm_posttraining.ask --model mariogemoll/Qwen2.5-Math-1.5B-GSM8K-SFT-LoRA "Janet has 5 apples..."
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


def _is_lora_adapter(model_id: str) -> bool:
    """Check whether a model path or HF repo contains a LoRA adapter."""
    local = Path(model_id)
    if local.is_dir():
        return (local / "adapter_config.json").exists()
    # Remote HF repo — check via the hub API
    from huggingface_hub import HfApi

    api = HfApi()
    siblings = api.model_info(model_id).siblings or []
    return any(s.rfilename == "adapter_config.json" for s in siblings)


def _load_adapter_base_model(model_id: str) -> str:
    """Read the base model ID from a LoRA adapter's config."""
    local = Path(model_id)
    if local.is_dir():
        cfg = json.loads((local / "adapter_config.json").read_text())
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, "adapter_config.json")
        cfg = json.loads(Path(path).read_text())
    base_path = cfg["base_model_name_or_path"]
    # Resolve relative local paths against the repo root
    if Path(base_path).exists() is False and not base_path.startswith("/"):
        candidate = _REPO_ROOT / base_path
        if candidate.exists():
            return str(candidate)
    return base_path


def load_model_and_tokenizer(model_id: str | None) -> tuple:
    """Load model and tokenizer.

    - No model: base Qwen model from HuggingFace.
    - LoRA adapter (local dir or HF repo): base model is read from the
      adapter config's base_model_name_or_path.
    - Plain model (local dir or HF repo): loaded directly.
    """
    if model_id is None:
        model_id = MODEL_ID
        print(f"Loading base model {model_id} ...", file=sys.stderr)
        tokenizer = _load_tokenizer(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tokenizer

    if _is_lora_adapter(model_id):
        base_path = _load_adapter_base_model(model_id)
        print(f"Loading base model from {base_path} ...", file=sys.stderr)
        tokenizer = _load_tokenizer(model_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        print(f"Loading LoRA adapter from {model_id} ...", file=sys.stderr)
        model = PeftModel.from_pretrained(base, model_id)
    else:
        print(f"Loading model from {model_id} ...", file=sys.stderr)
        tokenizer = _load_tokenizer(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
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
        "--model",
        default=None,
        help="Local path or HF repo ID (plain model or LoRA adapter). Omit for base Qwen.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    model, tokenizer = load_model_and_tokenizer(args.model)

    prompt = PROMPT_TEMPLATE.format(question=question)
    print(f"\n{GREY}{prompt}{RESET}", flush=True)
    chat_streaming(question, model, tokenizer, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
