# LLM posttraining

Small experiments for posttraining a math-focused language model on GSM8K.

The repository currently includes:

- supervised fine-tuning (SFT) on GSM8K chain-of-thought targets
- reinforcement learning with verifiable rewards (GRPO / RLVR)
- evaluation for base models, merged checkpoints, and LoRA adapters
- interactive one-turn chat for testing models and checkpoints
- a small Flask viewer web app backed by SQLite for inspecting training runs

The default base model is `Qwen/Qwen2.5-Math-1.5B`.

Recommended workflow:

1. `pip install -e .`
2. `python -m llm_posttraining.eval --backend vllm --split val`
3. `python -m llm_posttraining.sft`
4. `python -m llm_posttraining.eval --backend vllm --split val --ckpt checkpoints_sft/merged`
5. `python -m llm_posttraining.rlvr --base_model checkpoints_sft/merged`
6. `python -m llm_posttraining.eval --backend vllm --split val --ckpt checkpoints_rlvr/final`

## Project layout

- `src/llm_posttraining/sft.py`: SFT warm-up that trains a LoRA adapter and saves a merged model
- `src/llm_posttraining/rlvr.py`: GRPO training loop with reward logging
- `src/llm_posttraining/eval.py`: evaluation on GSM8K with Hugging Face or vLLM backends
- `src/llm_posttraining/ask.py`: one-turn streaming chat for interacting with a model or checkpoint
- `src/llm_posttraining/viewer.py`: local web UI for browsing logged runs
- `src/llm_posttraining/run_logger.py`: SQLite logger for runs, steps, and completions

## Setup

Install the package in editable mode:

```bash
pip install -e .
```

Install development tools too:

```bash
pip install -e ".[dev]"
```

Install the optional vLLM dependency if you want to use the vLLM backend:

```bash
pip install -e ".[vllm]"
```

## Training workflow

### 1. Baseline eval

Run an initial evaluation on the validation split before training:

```bash
python -m llm_posttraining.eval --split val
```

Use `test` only for the final report, not for repeated iteration.

### 2. Run SFT warm-up

This stage formats GSM8K reasoning traces so answers end in `\boxed{...}` and trains a LoRA adapter
before merging it back into the base model.

```bash
python -m llm_posttraining.sft
```

Output:

- `checkpoints_sft/`
- `checkpoints_sft/merged/` for the merged model, ready for GRPO or evaluation

Evaluate the SFT model:

```bash
python -m llm_posttraining.eval --split val --ckpt checkpoints_sft/merged
```

### 3. Run RLVR / GRPO

This stage uses an exact-match numeric reward on the answer found inside `\boxed{...}`.

```bash
python -m llm_posttraining.rlvr --base_model checkpoints_sft/merged
```

Evaluate the RLVR checkpoint on validation:

```bash
python -m llm_posttraining.eval --split val --ckpt checkpoints_rlvr/checkpoint-200
```

After you have chosen the final RLVR checkpoint on the validation split, run test-set evaluation
once at the end.

### 4. Final evaluation

These commands are for final reporting after all development decisions are done:

```bash
# final evaluation only
python -m llm_posttraining.eval --split test
python -m llm_posttraining.eval --split test --ckpt checkpoints_sft/merged
python -m llm_posttraining.eval --split test --ckpt checkpoints_rlvr/final
```

## Evaluation

The evaluator supports `val` and `test`, and can run on `cuda`, `mps`, or `cpu`. It also supports
merged checkpoints, LoRA checkpoints, and an optional vLLM backend when you need it.

## Publishing Adapters

If you trained RLVR on top of `checkpoints_sft/merged`, the final RLVR adapter is relative to that
merged SFT model rather than directly to Qwen. To export a single Qwen-relative adapter for
publishing, run:

```bash
python -m llm_posttraining.export_combined_adapter \
  --sft_merged checkpoints_sft/merged \
  --rlvr_adapter checkpoints_trl/final \
  --output_dir checkpoints_final_qwen_adapter
```

This reconstructs the final weights from `checkpoints_sft/merged + RLVR` and projects them back
into a fresh rank-16 LoRA adapter on top of `Qwen/Qwen2.5-Math-1.5B`.

## Interactive chat

Use `ask` to send a one-turn math question to the base model or any checkpoint. Responses are
streamed token-by-token, and answers inside `\boxed{...}` are highlighted.

```bash
# Base Qwen model
python -m llm_posttraining.ask "Janet has 5 apples..."

# A specific RLVR checkpoint
python -m llm_posttraining.ask --checkpoint checkpoints_rlvr/checkpoint-1200 "Janet has 5 apples..."

# The merged SFT model
python -m llm_posttraining.ask --checkpoint checkpoints_sft/merged "Janet has 5 apples..."
```

Use `--max-new-tokens` to control the generation length (default: 512).

## Viewer

Training runs are logged to `runs.db` in SQLite WAL mode. The viewer web app reads that database and
shows:

- run-level summaries
- per-step metrics such as reward, KL, loss, truncation, and timing
- grouped completions and extracted answers for each step

Start the local viewer:

```bash
python -m llm_posttraining.viewer
```

By default it serves `runs.db` on `http://127.0.0.1:5000`.

Use a different database or port if needed:

```bash
python -m llm_posttraining.viewer --db runs.db --port 8000
```
