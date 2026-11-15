# Reproduce Frugal Analyst

Run every command from the repo root. Run times and costs are from our own jobs (job list of 2026-11-14, plus the three `grpo-long` jobs from a teammate's account). Costs are estimates from run time and the flavor price; the billing page may differ, because it also bills start-up time.

## 1. Environment (local, no GPU)

Python 3.12 and `uv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pytest auditkit gradio pandas
uv pip install --python .venv/bin/python -r requirements-train.txt   # only for local training or inference
```

`requirements-train.txt` installs AgentTune at commit `1dfd054ce59e1c1bdd566cb7e5826f836b1c8c2e`. It resolved to torch 2.11.0, transformers 5.23.0, trl 1.7.1, peft 0.21.3, datasets 5.2.0 and accelerate 1.15.0. AuditKit is 1.1.11.

The base model `CohereLabs/tiny-aya-global` is gated. Accept its terms on Hugging Face, then export a token:

```bash
export HF_TOKEN=<your token>
```

## 2. Data and tests (local, free)

```bash
.venv/bin/python scripts/gen_data.py --seed 0
.venv/bin/python -m pytest -q
```

`gen_data.py` builds `data/dbs/*.sqlite`, the task splits in `data/tasks/`, the teacher traces in `data/teacher/`, and the `oracle` and `naive` traces in `outputs/traces/`. Two runs give identical files. The test suite has 411 tests. They passed in 35 s on our Mac.

## 3. GPU jobs on Hugging Face Jobs

Every GPU step runs through `scripts/jobs/frugal_job.py`. It downloads the code, installs AgentTune, runs `gen_data.py --seed 0`, runs one step and uploads the step folder (adapter, traces, `job.log`, `job_info.txt`) to an output repo. One-time setup (code repo, output repo, token) is in `scripts/jobs/README.md`.

Our runs used `--code-repo prathameshk421/frugal-analyst-code --output-repo prathameshk421/frugal-analyst-runs`. Replace them with your own:

```bash
COMMON="--code-repo YOU/frugal-analyst-code --output-repo YOU/frugal-runs"
```

Two flag traps:

- Pass step arguments as one `--extra "..."` string. `hf jobs uv run` drops a bare `--`. Our first SFT job failed this way.
- A single flag that starts with `--` needs the `=` form: `--extra=--no-stop-after-tool-call`. Our first try with `--extra --no-stop-after-tool-call` failed.

We did not record the `--timeout` values. The values below follow `scripts/jobs/README.md`. A job killed by its timeout does not upload, so keep them generous.

Never start a job without the team's go-ahead. Run a smoke job first.

### 3a. Smoke jobs (a10g-small)

```bash
hf jobs uv run --flavor a10g-small --timeout 40m --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step baseline --smoke --run-name smoke-baseline $COMMON
hf jobs uv run --flavor a10g-small --timeout 40m --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step sft --smoke --run-name smoke-sft $COMMON
hf jobs uv run --flavor a10g-small --timeout 40m --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step grpo --smoke --run-name smoke-grpo \
    --init-adapter YOU/frugal-runs:smoke-sft/adapter $COMMON
```

Ours took 2 min 30 s, 3 min 14 s and 2 min 45 s (about 0.04 to 0.05 USD each).

### 3b. Baselines (a10g-small)

```bash
# zero-shot: the headline baseline, turn not stopped after a tool call
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step baseline --run-name zero-shot-v2 --trace-label zero-shot \
    $COMMON --extra=--no-stop-after-tool-call

# zero-shot-stop: negative result, turn stopped after a tool call
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step baseline --run-name zero-shot-stop --trace-label zero-shot-stop $COMMON
```

| Run | Our time | Est. USD |
|---|---|---|
| zero-shot-v2 | 46 min 54 s | 0.78 |
| zero-shot-stop | 28 min 16 s | 0.47 |

### 3c. SFT (a10g-small)

```bash
# distilled-all: all 400 train tasks (reference only)
hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step sft --run-name sft $COMMON \
    --extra "--epochs 2 --lr 2e-4 --lora-r 16 --batch-size 2 --grad-accum 8"

# distilled: 274 tasks, no RL-only template
hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step sft --run-name sft-rlsplit $COMMON \
    --extra "--teacher data/teacher/train_sft.jsonl --epochs 2 --lr 2e-4 --lora-r 16 --batch-size 2 --grad-accum 8"

# distilled-mixed: same 274 tasks, mixed teacher (one unneeded preview on half the traces, added on purpose)
hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step sft --run-name sft-mixed $COMMON \
    --extra "--teacher data/teacher/train_sft_mixed.jsonl --epochs 2 --lr 2e-4 --lora-r 16 --batch-size 2 --grad-accum 8"
```

| Run | Our time | Est. USD |
|---|---|---|
| sft | 48 min 20 s | 0.81 |
| sft-rlsplit | 33 min 40 s | 0.56 |
| sft-mixed | 41 min 49 s | 0.70 |

### 3d. GRPO

```bash
# grpo: gated reward, 50 steps, L40S
hf jobs uv run --flavor l40sx1 --timeout 4h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step grpo --run-name grpo-rlsplit \
    --init-adapter YOU/frugal-runs:sft-rlsplit/adapter $COMMON \
    --extra "--tasks data/tasks/train_rl.jsonl --max-steps 50 --num-generations 4 --batch-size 4 --grad-accum 4 --lr 2e-5 --beta 0 --temperature 1.0 --scale-rewards batch --epsilon-high 0.28 --max-completion-length 256 --max-tool-steps 4 --reward-mode gated --stop-if-no-signal 3"

# grpo-ungated: ablation, ungated reward, 30 steps, A10G
hf jobs uv run --flavor a10g-small --timeout 4h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step grpo --run-name grpo-ungated \
    --init-adapter YOU/frugal-runs:sft-rlsplit/adapter $COMMON \
    --extra "--tasks data/tasks/train_rl.jsonl --max-steps 30 --num-generations 4 --batch-size 4 --grad-accum 4 --lr 2e-5 --beta 0 --temperature 1.0 --scale-rewards batch --epsilon-high 0.28 --max-completion-length 256 --max-tool-steps 4 --reward-mode ungated"

# grpo-mixed: gated reward, 30 steps, L40S (the A10G run stopped with an error after 23 min)
hf jobs uv run --flavor l40sx1 --timeout 4h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step grpo --run-name grpo-mixed \
    --init-adapter YOU/frugal-runs:sft-mixed/adapter $COMMON \
    --extra "--tasks data/tasks/train_rl.jsonl --max-steps 30 --num-generations 4 --batch-size 4 --grad-accum 4 --lr 2e-5 --beta 0 --temperature 1.0 --scale-rewards batch --epsilon-high 0.28 --max-completion-length 256 --max-tool-steps 4 --reward-mode gated"
```

| Run | Flavor | Our time | Est. USD |
|---|---|---|---|
| grpo-rlsplit | l40sx1 (1.80 USD/h) | 2 h 14 min | 4.04 |
| grpo-ungated | a10g-small | 1 h 53 min | 1.90 |
| grpo-mixed (failed try) | a10g-small | 23 min 39 s | 0.39 |
| grpo-mixed | l40sx1 | 1 h 33 min | 2.81 |

The second GRPO stage `grpo-long` (the final model) trains 150 more steps on top of the merged SFT and `grpo` adapters. We ran it on a teammate's account, so its init adapters were the public repos. Use your own `$COMMON` and adapter paths.

```bash
# grpo-long: gated reward, 150 more steps, lr 5e-5, L40S
hf jobs uv run --flavor l40sx1 --timeout 12h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step grpo --run-name grpo-long \
    --init-adapter prathameshk421/frugal-analyst-tiny-aya-sft:.,prathameshk421/frugal-analyst-tiny-aya-grpo:. $COMMON \
    --extra "--tasks data/tasks/train_rl.jsonl --max-steps 150 --num-generations 4 --batch-size 4 --grad-accum 4 --lr 5e-5 --beta 0 --temperature 1.0 --scale-rewards batch --epsilon-high 0.28 --max-completion-length 256 --max-tool-steps 4 --reward-mode gated --stop-if-no-signal 3"
```

| Run | Flavor | Our time | Est. USD |
|---|---|---|---|
| grpo-long | l40sx1 | 22,780 s (6 h 20 min) | 11.39 |

We took the flag values from the team's training notes for this run. We did not check `--stop-if-no-signal` and `--timeout` for it. The script default for the first is 3. The smoke job for this stage took about 116 s on an A10G (0.03 USD).

Our job list stores the `grpo-ungated` and `grpo-mixed` command lines cut after `--rew`. The flags before that point are the ones shown. The reward mode comes from the run name, and the training log agrees: for `grpo-ungated`, mean reward 0.736 equals `0.5 * correct + 0.5 * efficiency` (0.531 and 0.942). We do not know if `--stop-if-no-signal` was given for these two runs. Its default is 3.

### 3e. Evaluation (a10g-small, both test splits, 220 tasks)

The GRPO adapters train on top of the merged SFT adapter, so evaluation merges both, in order.

```bash
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-distilled-all-v2 --trace-label distilled-all \
    --adapters YOU/frugal-runs:sft/adapter $COMMON
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-distilled-rlsplit --trace-label distilled \
    --adapters YOU/frugal-runs:sft-rlsplit/adapter $COMMON
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-grpo-rlsplit --trace-label grpo \
    --adapters YOU/frugal-runs:sft-rlsplit/adapter,YOU/frugal-runs:grpo-rlsplit/adapter $COMMON
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-grpo-ungated --trace-label grpo-ungated \
    --adapters YOU/frugal-runs:sft-rlsplit/adapter,YOU/frugal-runs:grpo-ungated/adapter $COMMON
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-distilled-mixed --trace-label distilled-mixed \
    --adapters YOU/frugal-runs:sft-mixed/adapter $COMMON
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-grpo-mixed --trace-label grpo-mixed \
    --adapters YOU/frugal-runs:sft-mixed/adapter,YOU/frugal-runs:grpo-mixed/adapter $COMMON

# grpo-long: the final model. Merge three adapters in order: SFT, grpo, grpo-long.
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
    scripts/jobs/frugal_job.py --step eval --run-name eval-grpo-long --trace-label grpo-long \
    --adapters prathameshk421/frugal-analyst-tiny-aya-sft:.,prathameshk421/frugal-analyst-tiny-aya-grpo:.,aarush323/frugal-analyst-runs:grpo-long/adapter $COMMON
```

The run name `eval-grpo-long` is our suggestion. We did not check the name of the teammate's job.

| Run | Our time | Est. USD |
|---|---|---|
| eval-distilled-all-v2 | 23 min 14 s | 0.39 |
| eval-distilled-rlsplit | 23 min 50 s | 0.40 |
| eval-grpo-rlsplit | 23 min 56 s | 0.40 |
| eval-grpo-ungated | 26 min 42 s | 0.45 |
| eval-distilled-mixed | 26 min 40 s | 0.44 |
| eval-grpo-mixed | 26 min 33 s | 0.44 |
| eval-grpo-long | 21 min 46 s (1,306 s) | 0.36 |

Every evaluation uses greedy decoding, at most 4 executed tool calls, 256 new tokens per turn and seed 0. Each trace file gets a `.run.json` file with these settings.

### Total

32 jobs, 20.28 GPU hours, about 28.43 USD. This includes 9 smoke jobs, 5 failed jobs (2 of them smoke jobs) and 2 cancelled jobs.

Two accounts paid. The account `prathameshk421` paid for 29 jobs (13.56 hours, 16.64 USD). The teammate account `aarush323` paid for the 3 `grpo-long` jobs (6.72 hours, about 11.8 USD): training 11.39 USD, evaluation 0.36 USD and one smoke job 0.03 USD. Each account stayed under the 20 USD ceiling. The combined cost is above it.

The first main pipeline (zero-shot-v2, sft-rlsplit, grpo-rlsplit and their two evaluations) is about 6.18 USD. With the `grpo-long` stage, the pipeline for the final model is about 18.0 USD.

## 4. Get the traces back

```bash
hf download YOU/frugal-runs --local-dir outputs/hub-runs
cp outputs/hub-runs/*/traces/*.jsonl outputs/hub-runs/*/traces/*.run.json outputs/traces/
```

The `grpo-long` traces are in the teammate's output repo. Download and copy them the same way:

```bash
hf download aarush323/frugal-analyst-runs --local-dir outputs/hub-runs-long
cp outputs/hub-runs-long/*/traces/*.jsonl outputs/hub-runs-long/*/traces/*.run.json outputs/traces/
```

The evaluation scripts hash every input file. Compare the hashes in `docs/results/*.md` (section "Inputs") with your files.

## 5. Scoring (local, CPU, free)

```bash
# all models against zero-shot (docs/results/results.md)
.venv/bin/python scripts/eval_auditkit.py --traces outputs/traces/*.jsonl --out outputs/eval/

# tool-call accuracy against the oracle calls (docs/results/agent_results.md)
.venv/bin/python scripts/eval_agent.py --traces outputs/traces/*.jsonl --out outputs/eval/

# GRPO against SFT, the effect of RL alone (docs/results/results_vs_sft.md)
.venv/bin/python scripts/eval_auditkit.py --baseline distilled --out outputs/eval_vs_sft/ \
    --traces outputs/traces/zero-shot_test_*.jsonl outputs/traces/distilled_test_*.jsonl \
             outputs/traces/grpo_test_*.jsonl outputs/traces/grpo-ungated_test_*.jsonl \
             outputs/traces/grpo-long_test_*.jsonl

# mixed-teacher experiment (docs/results/results_mixed.md)
.venv/bin/python scripts/eval_auditkit.py --baseline distilled-mixed --out outputs/eval_mixed/ \
    --traces outputs/traces/distilled-mixed_test_*.jsonl outputs/traces/grpo-mixed_test_*.jsonl

# reward audit on the val split (docs/results/reward_audit.md)
.venv/bin/python scripts/reward_audit.py

# optional: replay old trace files and score them with the current reward
.venv/bin/python scripts/rescore_traces.py --traces outputs/traces/*.jsonl --out outputs/rescored/ --allow-vm-drift
```

The trace sets for the second and third `eval_auditkit.py` calls follow the "Inputs" tables in `docs/results/results_vs_sft.md` and `docs/results/results_mixed.md`. Exit codes: 0 ok, 1 AuditKit failed samples, 2 bad input, 3 AuditKit and the plain-Python recomputation differ. All our runs exited with 0 failed samples and full agreement.

`--allow-vm-drift` is needed because SQLite VM-step counts differ slightly between machines (13 of 880 traces in our local replay). Rows and outputs must still match, or the script stops.

## 6. Demo (local, no GPU)

```bash
.venv/bin/python app/app.py     # http://127.0.0.1:7860
```

It replays the trace files in `outputs/traces/`, runs scripted policies live and has a SQL playground with VM-step cost. A live model appears only with `FRUGAL_LIVE_MODEL=1` and the training dependencies installed (`app/README.md`).
