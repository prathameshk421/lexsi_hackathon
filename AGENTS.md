# AGENTS.md

Instructions for AI coding agents in this repo. Check facts on disk before you change them.

## 1. Project

Frugal Analyst is our entry for the Lexsi AI Safety and Alignment Hackathon, Track 3 (Agentic Fine-tuning). Team Dear_Basketball has 3 beginners, so keep code simple.

The idea: train Tiny Aya to answer analytics questions over SQLite with tools, using few tool calls, few rows and low engine work.
Pipeline: zero-shot baseline, SFT on teacher traces, GRPO with a gated reward. AuditKit scores all models.

`docs/spec.md` is the contract for all code. If code and spec disagree, fix one of them in the same change.
`docs/plan.md` describes an older reconciliation idea. Keep it for history only.

## 2. Hard constraints

- The core model is a Cohere Labs open-weights model: `CohereLabs/tiny-aya-global`. It is gated. Accept its terms first. Other models are allowed only as teacher, validator or judge.
- AgentTune is required for training. AuditKit is required for evaluation. Use both in the real pipeline.
- If a toolkit has a bug, report it to a human. Do not replace the toolkit silently.
- Deadline: Nov 15, 2026, 11:59 PM AoE. That is Monday Nov 16, 2026, 5:29 PM IST. Only the last submission counts.
- Compute: Hugging Face Jobs. Budget ceiling is 20 USD. Spend only what the work needs.
- Run a smoke job before any real job. Never start a GPU job unless a human says so.
- All HF models, adapters, datasets, logs and traces are public and un-gated.
- Work must be our own, written during the hackathon.

## 3. Deliverables

- Public GitHub repo with reproduction steps. Write-up: HF Article or README.
- More than one before/after metric on the same model: task success, tool-call accuracy, average latency.
- Two public HF collections: one for models and adapters, one for datasets, logs and traces.
- Abstract under 500 words. Optional ROI table.
- Every write-up number traces to a saved run (config, seed, trace file). A number with no source is removed.

## 4. Layout (real tree)

```
frugal_analyst/           core package, standard library only
  db.py                   synthetic SQLite databases, one per domain, seeded
  tasks.py                Task dataclass, task generator, splits, jsonl io
  env.py                  tools get_schema and run_sql, read-only metered SQL
  prompts.py              SYSTEM_PROMPT, build_user_prompt(task)
  reward.py               score_trajectory(), answer matching, RewardConfig
  policies.py             scripted policies: oracle, naive, guess (no model)
  traces.py               trace schema, jsonl io, run_scripted_policy()
  metrics.py              per-episode metrics, aggregate, paired bootstrap CI
  trajectory.py           parse AgentTune completion text
  runner.py               run a HF model with optional LoRA, write traces
scripts/
  gen_data.py             build databases, tasks and teacher traces
  eval_auditkit.py        AuditKit evaluation over trace files
  rescore_traces.py       replay old trace files, score them with the current reward
  reward_audit.py         mean reward of honest and cheating behaviours, val split only
  train_sft.py            distillation with AgentTune
  train_grpo.py           GRPO with AgentTune
  jobs/                   HF Jobs entry point and README
app/
  app.py                  Gradio demo, also the HF Space entry point
  fa_*.py, README.md, requirements.txt   demo helpers, docs, deps
tests/                    pytest files, one per module group
data/tasks/, data/teacher/ task and teacher jsonl (committed)
data/dbs/                 SQLite files, git-ignored, rebuilt by gen_data.py
docs/                     spec.md (contract), eval-notes.md, training-notes.md, other plan notes
outputs/                  runs, traces, eval results, HF cache. git-ignored
pyproject.toml, requirements-train.txt   package and pytest config; AgentTune pin and training deps
```

## 5. Commands

Verified commands, from the repo root:

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pytest auditkit gradio pandas
uv pip install --python .venv/bin/python -r requirements-train.txt   # training only
.venv/bin/python -m pytest -q
.venv/bin/python scripts/gen_data.py --seed 0
.venv/bin/python scripts/eval_auditkit.py --traces outputs/traces/*.jsonl --out outputs/eval/
.venv/bin/python scripts/rescore_traces.py --traces outputs/traces/*.jsonl --out outputs/rescored/ --allow-vm-drift
.venv/bin/python scripts/reward_audit.py                             # writes outputs/eval/reward_audit.md
.venv/bin/python app/app.py                                          # http://127.0.0.1:7860
```

Training and HF Jobs commands: see `scripts/jobs/README.md`. Do not launch a job without a human's go-ahead.
Pass step arguments to `frugal_job.py` as one `--extra "..."` string. `hf jobs uv run` drops a bare `--`.
If the string is one flag with no value, write `"--extra=--no-stop-after-tool-call"`. Without the `=`, argparse reads it as a new option and the job fails.
Use `--trace-label`, not `--label`: `hf jobs uv run` takes `--label` for itself.

## 6. Toolkit facts

- AgentTune installs from source at commit `1dfd054`. It needs Python 3.12 or later.
- AgentTune's `TrlAgenticGrpo` drops unknown kwargs silently. Print the resolved `GRPOConfig`. `train_grpo.py` stops on a mismatch.
- In agentic GRPO the reward gets each completion as one flat string. Our reward reads the `trajectories` kwarg first and parses text only as a fallback.
- AuditKit 1.1.11: `ak.evaluate(model="precomputed", cache=False)` per (model, split), `ak.compare` for pairs, `ak.scorer` for metrics. Do not use `ak.compare_models`: it cannot give each model its own precomputed output.
- AuditKit has no confidence interval. Our CI is `metrics.paired_bootstrap_ci`: paired, task resampling, fixed seed. `ak.compare(...).significance()` gives AuditKit's own p-value. Label both.
- A non-zero `failed_count` makes an evaluation run invalid. Exit codes: 0 ok, 1 failed samples, 2 bad input, 3 cross-check mismatch.

## 7. Working rules

- Verify, do not guess. Check the installed package or run a one-line test before you use an API.
- Report results as they are. Never tune a metric or the data to look better.
- Every experiment uses a fixed seed and a saved config. No data leaks. Test splits and the held-out `saas` domain never enter training. The test set never tunes the reward.
- A reward change needs tests. A perfect answer scores 1.0. An empty answer scores 0. An answer with no query scores at most 0.05 in gated mode.
- A correct answer needs evidence: a cell from a query that read a data table, and not a literal in any select list of that query. Run `scripts/reward_audit.py` after a reward change.
- Do not weaken SQL tool security: read-only, SELECT or WITH only, one statement, row cap, output cap, VM-step abort.
- Escape all model and database text before it appears in the app. Ask before you add a dependency or a top-level folder.
- Never commit secrets. Read them from environment variables, for example `HF_TOKEN`.
- No model weights or large data in git. They go to Hugging Face.

## 8. Git

- Work on a branch. Never commit to `main`.
- Do not add yourself as author, committer or co-author. Add no `Co-Authored-By` lines.
- No "Generated with" lines and no tool names in commits, PRs or code comments. Use the git identity already set on the machine. Do not change `git config`.
- Commit only when a human asks, or has said to commit freely for the current task. Never force-push, rewrite history or delete branches unless a human asks.
- Subject: one imperative line, under 60 characters. Body (optional) says why. Plain words. No emoji, no em dashes.
- Before each commit, run `git status` and `git diff`. Do not stage files you did not change.

## 9. Keep this file up to date

- Before you finish a task, check whether it made anything here wrong or incomplete.
- If it did, edit this file on the same branch, in the same commit. Change only what is wrong, and write only checked facts.
- If nothing changed, do not touch this file.
- Claude Code reads this file only when no `CLAUDE.md` or `CLAUDE.local.md` exists. Do not add one. If you must, its first line is `@AGENTS.md`.



## 10. Writing style (README, write-up, docs, comments, commits)

- Use short sentences, one idea each, active voice, plain words.
- Use numbers instead of adjectives: "40 times faster (21 s to 0.5 s)".
- Avoid marketing words (robust, seamless, cutting-edge, leverage). No em dashes.
- Comments explain why, not what.
