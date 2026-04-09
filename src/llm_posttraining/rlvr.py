# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""GRPO training on GSM8K using TRL's GRPOTrainer.

Trains the model with Group Relative Policy Optimization (GRPO) using
a verifiable reward: 1.0 if the answer in \\boxed{} matches, else 0.0.

Usage:
    python -m llm_posttraining.rlvr
    python -m llm_posttraining.rlvr --use_vllm
    python -m llm_posttraining.rlvr --base_model checkpoints_sft/merged --use_vllm

TensorBoard:
    tensorboard --logdir checkpoints_rlvr
"""

import argparse
import importlib.util
import math
import os
import re
import time

import torch
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl import GRPOConfig, GRPOTrainer

from llm_posttraining.data import PROMPT_TEMPLATE, load_gsm8k
from llm_posttraining.model import LORA_CONFIG, MAX_SEQ_LEN, MODEL_ID, load_tokenizer
from llm_posttraining.reward import compute_reward, extract_answer
from llm_posttraining.run_logger import RunLogger


def resolve_attn_implementation(attn_implementation: str) -> str:
    """Pick the best supported attention backend for this machine."""
    if attn_implementation != "auto":
        return attn_implementation
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


# ── Callbacks ────────────────────────────────────────────────────────────────


class StepStatsCallback(TrainerCallback):
    """Prints a one-line stats summary after every logging step and writes to RunLogger."""

    def __init__(self, run_logger: RunLogger, beta: float, completion_logger: "CompletionLogger"):
        self.run_logger = run_logger
        self.beta = beta
        self.cl = completion_logger
        self._step_start: float | None = None

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._step_start = time.time()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss", float("nan"))
        reward = logs.get("reward", logs.get("train/reward", float("nan")))
        kl = logs.get("kl", logs.get("train/kl", float("nan")))
        lr = logs.get("learning_rate", float("nan"))
        grad_norm = logs.get("grad_norm", float("nan"))

        pg_loss = (
            loss - self.beta * kl if not (math.isnan(loss) or math.isnan(kl)) else float("nan")
        )

        format_rate = self.cl.format_rates.get(step)
        avg_length = self.cl.avg_lengths.get(step)
        truncated = self.cl.truncated_counts.get(step)

        now = time.time()
        step_time = (now - self._step_start) if self._step_start is not None else None
        gen_end = self.cl.gen_end_times.get(step)
        gen_time = (
            (gen_end - self._step_start)
            if (gen_end is not None and self._step_start is not None)
            else None
        )

        print(
            f"  step {step:>5} | loss {loss:.4f} | pg_loss {pg_loss:.4f} | reward {reward:.4f}"
            f" | kl {kl:.4f} | lr {lr:.2e} | gnorm {grad_norm:.3f}"
            + (f" | fmt {format_rate:.0%}" if format_rate is not None else "")
            + (f" | trunc {truncated}" if truncated else "")
            + (f" | gen {gen_time:.1f}s/{step_time:.1f}s" if step_time is not None else "")
        )

        if step > 0:

            def _clean(v: float) -> float | None:
                return None if math.isnan(v) else v

            self.run_logger.log_step(
                step,
                loss=_clean(loss),
                pg_loss=_clean(pg_loss),
                reward=_clean(reward),
                kl=_clean(kl),
                lr=_clean(lr),
                truncated_count=truncated,
                grad_norm=_clean(grad_norm),
                format_rate=format_rate,
                avg_completion_length=avg_length,
                step_time=step_time,
                gen_time=gen_time,
            )


# ── Completion logging ───────────────────────────────────────────────────────


class CompletionLogger:
    """Wraps the reward function to capture completions into RunLogger's SQLite DB."""

    def __init__(self, run_logger: RunLogger, tokenizer, max_completion_length: int | None):
        self.run_logger = run_logger
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length
        self._step = 0
        self.truncated_counts: dict[int, int] = {}
        self.format_rates: dict[int, float] = {}
        self.avg_lengths: dict[int, float] = {}
        self.gen_end_times: dict[int, float] = {}

    def wrap(self, reward_fn):
        def wrapped(completions, answer, **kwargs):
            self._step += 1
            self.gen_end_times[self._step] = time.time()
            rewards = reward_fn(completions, answer, **kwargs)

            texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]

            # Per-completion stats
            token_lengths = [len(self.tokenizer.encode(t)) for t in texts]
            self.avg_lengths[self._step] = sum(token_lengths) / len(token_lengths)
            max_completion_length = self.max_completion_length
            n_truncated = (
                sum(length >= max_completion_length for length in token_lengths)
                if max_completion_length is not None
                else 0
            )
            self.truncated_counts[self._step] = n_truncated
            if n_truncated:
                print(f"  [truncated {n_truncated}/{len(texts)} completions at step {self._step}]")
            self.format_rates[self._step] = sum(
                1 for t in texts if re.search(r"\\boxed\{", t)
            ) / len(texts)

            # Extract prompt texts (TRL passes these in kwargs)
            raw_prompts = kwargs.get("prompts", [])
            if raw_prompts:
                prompt_texts = [
                    p[-1]["content"] if isinstance(p, list) else str(p) for p in raw_prompts
                ]
                if len(prompt_texts) < len(texts):
                    n = len(texts) // len(prompt_texts)
                    prompt_texts = [pt for pt in prompt_texts for _ in range(n)]
            else:
                prompt_texts = [None] * len(texts)

            self.run_logger.log_completions(
                self._step,
                prompts=prompt_texts,
                completions=texts,
                predicted=[extract_answer(t) for t in texts],
                expected=list(answer),
                rewards=list(rewards),
            )

            return rewards

        return wrapped


# ── Dataset & reward ─────────────────────────────────────────────────────────


def build_dataset(split) -> Dataset:
    """Format GSM8K examples as {prompt, answer} for GRPO."""
    rows = [
        {"prompt": PROMPT_TEMPLATE.format(question=ex["question"]), "answer": ex["answer"]}
        for ex in split
    ]
    return Dataset.from_list(rows)


def make_reward_fn():
    """Build a reward function compatible with TRL's chat-message format.

    TRL 0.29+ passes completions as list[list[dict]] (chat messages).
    Each completion is [{"role": "assistant", "content": "..."}].
    """

    def reward_fn(completions, answer, **kwargs):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [compute_reward(t, a) for t, a in zip(texts, answer)]

    return reward_fn


# ── Training ─────────────────────────────────────────────────────────────────


def train(
    output_dir: str = "checkpoints_rlvr",
    base_model: str | None = "checkpoints_sft/merged",
    use_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.7,
    attn_implementation: str = "auto",
    num_generations: int = 8,
    prompts_per_step: int = 2,
    epochs: int = 1,
    max_steps: int = -1,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data ...")
    splits = load_gsm8k()
    train_dataset = build_dataset(splits["train"])

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer()

    model_id = base_model or MODEL_ID
    print(f"Base model: {model_id}")

    resolved_attn_implementation = resolve_attn_implementation(attn_implementation)
    print(f"Attention backend: {resolved_attn_implementation}")

    steps_per_epoch = len(train_dataset) // prompts_per_step
    effective_max_steps = max_steps if max_steps > 0 else epochs * steps_per_epoch

    config = GRPOConfig(
        output_dir=output_dir,
        # rollout
        num_generations=num_generations,
        max_completion_length=MAX_SEQ_LEN,
        temperature=0.8,
        top_p=0.95,
        # training
        per_device_train_batch_size=num_generations * prompts_per_step,
        num_train_epochs=epochs,
        max_steps=effective_max_steps,
        learning_rate=1e-5,
        lr_scheduler_type="warmup_stable_decay",
        warmup_steps=50,
        lr_scheduler_kwargs={
            "num_stable_steps": max(
                1, effective_max_steps - 50 - max(100, effective_max_steps // 10)
            ),
            "num_decay_steps": max(100, effective_max_steps // 10),
        },
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        model_init_kwargs={
            "dtype": "bfloat16",
            "attn_implementation": resolved_attn_implementation,
        },
        # GRPO / KL
        loss_type="dr_grpo",
        scale_rewards="none",
        beta=0.1,
        # vLLM
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_importance_sampling_correction=False,
        # logging / saving
        logging_steps=1,
        save_steps=200,
        save_total_limit=None,
        save_only_model=True,
        report_to="tensorboard",
    )

    # Run logger (SQLite)
    run_logger = RunLogger(
        "trl",
        {
            "num_generations": config.num_generations,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "learning_rate": config.learning_rate,
            "beta": config.beta,
            "max_completion_length": config.max_completion_length,
            "max_steps": config.max_steps,
            "num_train_epochs": config.num_train_epochs,
            "temperature": config.temperature,
            "use_vllm": use_vllm,
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            "attn_implementation": resolved_attn_implementation,
            "base_model": model_id,
            "prompts_per_step": prompts_per_step,
        },
    )

    # Completion logger (wraps reward fn to capture completions)
    comp_logger = CompletionLogger(run_logger, tokenizer, config.max_completion_length)
    reward_fn = comp_logger.wrap(make_reward_fn())

    trainer = GRPOTrainer(
        model=model_id,
        args=config,
        peft_config=LORA_CONFIG,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        callbacks=[StepStatsCallback(run_logger, config.beta, comp_logger)],
    )

    try:
        print("Training ...")
        trainer.train()

        print(f"Saving final checkpoint to {output_dir}/final ...")
        trainer.save_model(f"{output_dir}/final")
        tokenizer.save_pretrained(f"{output_dir}/final")
        run_logger.finish("done")
    except Exception:
        run_logger.finish("failed")
        raise
    finally:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K")
    parser.add_argument("--output_dir", default="checkpoints_rlvr")
    parser.add_argument(
        "--base_model",
        default="checkpoints_sft/merged",
        help="Path to merged SFT model. Defaults to checkpoints_sft/merged.",
    )
    parser.add_argument(
        "--use_vllm", action="store_true", help="Use vLLM for fast rollout generation"
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.7,
        help="Fraction of GPU memory reserved for vLLM when --use_vllm is enabled.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
        help="Attention backend for the training model. auto prefers flash_attention_2 when available, else sdpa.",
    )
    parser.add_argument("--num_generations", type=int, default=8, help="Rollouts per prompt (G)")
    parser.add_argument(
        "--prompts_per_step",
        type=int,
        default=2,
        help="Prompts per optimizer step. batch_size = num_generations * prompts_per_step",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of passes through the training set"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Override max training steps (-1 = auto)"
    )
    args = parser.parse_args()
    train(
        output_dir=args.output_dir,
        base_model=args.base_model,
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        attn_implementation=args.attn_implementation,
        num_generations=args.num_generations,
        prompts_per_step=args.prompts_per_step,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
