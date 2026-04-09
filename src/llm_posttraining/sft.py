# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""SFT warm-up on GSM8K chain-of-thought solutions.

Trains the model to produce step-by-step reasoning ending in \\boxed{N}
before GRPO, giving the policy a strong prior toward the correct format
so GRPO gets denser reward signal.

Usage:
    python -m llm_posttraining.sft
    python -m llm_posttraining.sft --output_dir checkpoints_sft --epochs 2

Output: checkpoints_sft/merged (LoRA weights merged into base, ready for GRPO)
"""

import argparse
import os

from datasets import Dataset
from peft import PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer

from llm_posttraining.data import PROMPT_TEMPLATE, format_chain, load_gsm8k
from llm_posttraining.model import LORA_CONFIG, MAX_SEQ_LEN, load_base_model, load_tokenizer


def build_dataset(split, tokenizer) -> Dataset:
    """Format GSM8K examples as prompt+chain text, filtering overlength ones."""
    rows = []
    skipped = 0
    total = 0
    for ex in split:
        total += 1
        chain = format_chain(ex["chain"], ex["answer"])
        text = PROMPT_TEMPLATE.format(question=ex["question"]) + chain
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens > MAX_SEQ_LEN:
            skipped += 1
            continue
        rows.append({"text": text})
    print(f"  Skipped {skipped}/{total} examples exceeding {MAX_SEQ_LEN} tokens ({skipped / total:.1%})")
    return Dataset.from_list(rows)


def train(output_dir: str = "checkpoints_sft", epochs: int = 1):
    print("Loading data ...")
    splits = load_gsm8k()

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer()

    train_dataset = build_dataset(splits["train"], tokenizer)
    print(f"  {len(train_dataset)} training examples")
    val_dataset = build_dataset(splits["val"], tokenizer)
    print(f"  {len(val_dataset)} val examples")

    print("Loading model ...")
    model = get_peft_model(load_base_model(), LORA_CONFIG)
    model.print_trainable_parameters()

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        max_length=MAX_SEQ_LEN,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("Training ...")
    trainer.train()

    print("Merging LoRA weights into base model ...")
    assert isinstance(trainer.model, PeftModel)
    merged = trainer.model.merge_and_unload()
    merged_path = os.path.join(output_dir, "merged")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Saved to {merged_path}")


def main():
    parser = argparse.ArgumentParser(description="SFT warm-up on GSM8K")
    parser.add_argument("--output_dir", default="checkpoints_sft")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    train(output_dir=args.output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
