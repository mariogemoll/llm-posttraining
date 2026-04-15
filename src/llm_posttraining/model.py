# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Model loading utilities and shared constants."""

import torch
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
MAX_SEQ_LEN = 384

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)


def load_tokenizer(model_id: str = MODEL_ID) -> PreTrainedTokenizerBase:
    """Load tokenizer with pad token set to eos if missing."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_id: str = MODEL_ID,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
) -> PreTrainedModel:
    """Load the base model for training or inference."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    assert isinstance(model, PreTrainedModel)
    return model
