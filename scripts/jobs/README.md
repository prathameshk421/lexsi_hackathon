# Hugging Face Jobs

`frugal_job.py` is one self-contained uv script for every GPU step. It downloads
this project, installs AgentTune at commit 1dfd054 (from its GitHub tarball, so
the image needs no git), runs `scripts/gen_data.py --seed 0`, runs one step, and
uploads the step folder (adapter, traces, `job.log`, `job_info.txt`) to the Hub
repo you name. It uploads in a `finally` block, so a failed step still leaves its log.

These commands ran for the submission. Measured run times and costs are in `submission/reproduce.md`. The cost estimate below was made before the runs.

## One-time setup (you do this)

1. Put the code on the Hub as a private dataset repo. The job downloads it with
   `snapshot_download`, skipping `.venv`, `outputs` and `data/dbs`.

   ```bash
   hf repos create YOU/frugal-analyst-code --repo-type dataset --private
   hf upload YOU/frugal-analyst-code . . --repo-type dataset \
       --exclude ".venv/*" --exclude "outputs/*" --exclude "data/dbs/*" --exclude ".git/*"
   ```

   A git URL also works (`--code-git-url URL`), but only if the job image has git.

2. Pick an output repo, for example `YOU/frugal-runs` (a private model repo).
   The first job creates it. Each job writes into its own folder `--run-name`.

3. The token. `--secrets HF_TOKEN` reads `HF_TOKEN` from your shell. It needs
   read access to `CohereLabs/tiny-aya-global` (gated) and write access to the output repo.

   ```bash
   export HF_TOKEN=$(cat ~/.cache/huggingface/token)
   ```

## Commands

All commands use `a10g-small` (1x A10G 24 GB, $1.00/h). Replace `YOU`.
Arguments after `--` go to the step's script (`--help` on each script lists them).

Common flags (copy into each command):

```bash
COMMON="--code-repo YOU/frugal-analyst-code --output-repo YOU/frugal-runs"
```

### 0. Smoke mode (a few cents per job)

Every step takes `--smoke`. It proves the image, the install, the gated download,
the step's code path and the upload with tiny settings:

| Step | `--smoke` adds |
|---|---|
| baseline, eval | `--limit 5`, test_id only |
| sft | `--limit 16 --max-steps 5` |
| grpo | `--limit 8 --max-steps 2 --num-generations 2 --batch-size 2 --grad-accum 1 --max-tool-steps 3` |

Arguments after `--` still override these. Most of a smoke job's time is the
fixed overhead (install plus 6.7 GB download, about 10 minutes), so expect
about $0.10 to $0.25 each. Run at least the baseline smoke first; read the
seconds per task in `job.log` to check the estimates below.

```bash
hf jobs uv run --flavor a10g-small --timeout 40m --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step baseline --smoke --run-name smoke-baseline $COMMON
hf jobs uv run --flavor a10g-small --timeout 40m --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step sft --smoke --run-name smoke-sft $COMMON
hf jobs uv run --flavor a10g-small --timeout 40m --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step grpo --smoke --run-name smoke-grpo \
    --init-adapter YOU/frugal-runs:smoke-sft/adapter $COMMON
```

Optional, cheaper evals: a T4 has no bf16, so the runner picks fp16 there.
Tiny Aya in fp16 is not verified. If you want to try `t4-small` ($0.40/h) for
the eval steps, run the baseline smoke on it first and compare its `job.log`
with the a10g-small smoke: same call format, no NaN or empty output. If it
looks wrong, stay on a10g-small.

### 1. Zero-shot baseline (both test splits, 220 tasks)

```bash
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step baseline --run-name zero-shot $COMMON
```

Writes `zero-shot/traces/zero-shot_test_id.jsonl` and `zero-shot_test_ood.jsonl`.

### 2. SFT (distillation warm start)

```bash
hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step sft --run-name sft $COMMON \
    --extra "--epochs 2 --lr 2e-4 --lora-r 16 --batch-size 2 --grad-accum 8"
```

Pass step arguments as one `--extra "..."` string. `hf jobs uv run` drops a bare `--`.

Writes the adapter to `sft/adapter/`.

### 3. GRPO (gated reward) on top of the SFT adapter

```bash
hf jobs uv run --flavor a10g-small --timeout 4h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step grpo --run-name grpo \
    --init-adapter YOU/frugal-runs:sft/adapter $COMMON \
    --extra "--max-steps 40 --num-generations 4 --batch-size 4 --grad-accum 4 --lr 1e-5 --beta 0.04 --temperature 0.8 --max-completion-length 256 --max-tool-steps 4 --reward-mode gated"
```

Writes the adapter to `grpo/adapter/`, plus `reward_log.jsonl` (mean reward per
step) and `trajectories.jsonl` (every rollout).

### 4. Evaluate the distilled and GRPO models

The GRPO adapter was trained on top of the merged SFT adapter, so evaluate it
with both, in order.

```bash
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-distilled --trace-label distilled \
    --adapters YOU/frugal-runs:sft/adapter $COMMON

hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-grpo --trace-label grpo \
    --adapters YOU/frugal-runs:sft/adapter,YOU/frugal-runs:grpo/adapter $COMMON
```

### 5. Optional ablation: ungated reward

Same as step 3 with `--run-name grpo-ungated` and `--reward-mode ungated`, then
step 4 with `--trace-label grpo-ungated` and `--adapters ...:sft/adapter,...:grpo-ungated/adapter`.

### 6. Pipeline with RL-only templates (use this one now)

The first GRPO run got reward 1.0 on every rollout and gradient 0: SFT had
seen every train template. `scripts/gen_data.py` now also writes:

| File | Content |
|---|---|
| `data/tasks/train_sft.jsonl`, `data/teacher/train_sft.jsonl` | 274 train tasks, no RL-only template |
| `data/tasks/train_rl.jsonl` | 168 tasks: all 126 RL-only train tasks plus 42 SFT-seen tasks to check for forgetting |

RL-only means the last template of each level in each train domain
(`RL_ONLY_TEMPLATES` in `frugal_analyst/tasks.py`). 38 of the 120 test_id tasks
come from those templates.

Smoke each step first (add `--smoke`, change `--run-name`).

```bash
# 6a. SFT on train_sft only
hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step sft --run-name sft-rlsplit $COMMON \
    --extra "--teacher data/teacher/train_sft.jsonl --epochs 2 --lr 2e-4 --lora-r 16 --batch-size 2 --grad-accum 8"

# 6b. GRPO on train_rl, on top of that adapter
hf jobs uv run --flavor a10g-small --timeout 4h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step grpo --run-name grpo-rlsplit \
    --init-adapter YOU/frugal-runs:sft-rlsplit/adapter $COMMON \
    --extra "--tasks data/tasks/train_rl.jsonl --max-steps 40 --num-generations 4 --batch-size 4 --grad-accum 4 --lr 1e-5 --beta 0.04 --temperature 0.8 --max-completion-length 256 --max-tool-steps 4 --reward-mode gated --stop-if-no-signal 3"

# 6c. Evaluations (all with the same harness: stop after a tool call, 4 tool steps)
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step baseline --run-name zero-shot-stop $COMMON
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-distilled-rlsplit --trace-label distilled \
    --adapters YOU/frugal-runs:sft-rlsplit/adapter $COMMON
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-grpo-rlsplit --trace-label grpo \
    --adapters YOU/frugal-runs:sft-rlsplit/adapter,YOU/frugal-runs:grpo-rlsplit/adapter $COMMON
```

Notes:

- No-signal guard. If no GRPO group has any reward spread in each of the first
  3 optimizer steps, `train_grpo.py` stops, saves no adapter and exits with
  code 4. The job then shows `status=4` in `job_info.txt` and still uploads
  `reward_log.jsonl` and `trajectories.jsonl`. `--stop-if-no-signal 0` turns it off.
- New GRPO flags, all real `GRPOConfig` fields in trl 1.7.1: `--top-p`
  (default 0.95, AgentTune's default), `--scale-rewards {group,batch,none}`
  (default group), `--loss-type` (default dapo), `--epsilon-high` (default:
  same as epsilon, 0.2). The script prints the resolved config and stops if
  one did not take effect. These are untested suggestions, not measured
  results: `--scale-rewards batch` (with group scaling, a gap of 0.01 in
  efficiency inside a group becomes a full-size advantage) and
  `--epsilon-high 0.28` (DAPO's value).
- Evaluation harness. The runner now ends a model turn as soon as it ends with
  a complete tool call, for every model. `--extra "--no-stop-after-tool-call"`
  gives the old behaviour. Each trace file gets a `<name>.run.json` next to it
  with the harness settings. The runner default for `--max-tool-steps` is now 4,
  the same as GRPO (it was 5).

### Get the results back

```bash
hf download YOU/frugal-runs --local-dir outputs/hub-runs
cp outputs/hub-runs/*/traces/*.jsonl outputs/hub-runs/*/traces/*.run.json outputs/traces/
```

## Cost estimate

Prices: t4-small $0.40/h (T4 16 GB), l4x1 $0.80/h (L4 24 GB), a10g-small $1.00/h
(A10G 24 GB), a10g-large $1.50/h. Budget: $20.

Assumptions (not measured; the smoke job checks them):

- Job overhead: image start, uv install of AgentTune's dependency set, and the
  6.7 GB model download. About 10 minutes per job.
- Tiny Aya decode with plain `transformers.generate`, batch 1, bf16 on A10G:
  about 25 tokens/s. Each episode is 3 to 4 model turns of about 60 to 100
  tokens when the model is on format. Zero-shot Tiny Aya on this Mac used 116
  to 398 generated tokens per episode (3 episodes), because it writes a fake
  "Tool result" after each call. The worst case is 6 turns x 256 tokens.
- SFT: 1,346 rows (254 three-turn and 146 four-turn teacher episodes), about
  1,300 tokens each, so about 1.75 M tokens per epoch, at about 1,300 tokens/s
  with LoRA and gradient checkpointing.
- GRPO: 16 rollouts per optimizer step, generated one at a time, plus one
  training pass over 16 sequences of about 1,700 tokens.

| Step | Flavor | Time (typical, worst) | Cost (typical, worst) |
|---|---|---|---|
| 0. Smoke (10 tasks) | a10g-small | 15 min | $0.25 |
| 1. Zero-shot eval, 220 tasks | a10g-small | 1 h, 3 h | $1.00, $3.00 |
| 2. SFT, 2 epochs | a10g-small | 55 min, 1.5 h | $0.95, $1.50 |
| 3. GRPO, 40 steps | a10g-small | 2.3 h, 4 h | $2.30, $4.00 |
| 4. Eval distilled + GRPO | a10g-small | 2 x 45 min | $1.50, $3.00 |
| Subtotal, main pipeline | | | $6.00, $11.75 |
| 5. Ablation GRPO + eval | a10g-small | 3 h | $3.00, $5.00 |

The `--timeout` values above are set to the worst case so a stuck job cannot
burn the budget. A job killed by the timeout does not reach the upload, so its
outputs are lost; raise the timeout rather than cut it close.

Why a10g-small: decoding one sequence at a time is limited by memory bandwidth.
The A10G has about twice the bandwidth of the L4 (600 vs 300 GB/s), so
generation-heavy steps (eval, GRPO) should cost less on a10g-small than on l4x1
even at $0.20/h more. SFT alone would be about the same on either. The T4 has no
bf16, and 16 GB is tight for GRPO with a 262k-token vocabulary, so avoid t4-small.
