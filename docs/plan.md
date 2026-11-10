# Lexsi hackathon: honest review of the reconciliation agent, and a revised plan

## Context

Team of 3 beginners, Agentic Fine-tuning track, problem statement 6.
Deadline: Nov 15, 5:29 PM IST. About 4.5 days remain from Nov 10.
You asked for an honest review, the weak points, and a stronger plan.
Sources: your research PDF, hackathon.lexsi.ai, agenttune.lexsi.ai, auditkit.lexsi.ai (all read on Nov 10).

## Verdict

The idea is good. The plan in the doc is not yet a plan that wins.
The domain and the exact ground truth are real strengths. Keep them.
The doc misses three facts that decide the result. Fix those first.
I do not have a better idea that you can also finish in 4.5 days.

## Three facts the doc misses

1. **The base model is fixed.** The rules say the core submission must use a Cohere Labs open-weights model.
   The list: Tiny Aya (Base, Global, Fire, Water, Earth), Aya Expanse 8B, Aya Expanse 32B.
   Other models are allowed only as teacher, validator, or judge.
   Tiny Aya is a small multilingual model. AgentTune uses a prompt-based tool fallback for Cohere models.
   A multi-step task across 3 systems with a paginated API is very hard for this model.
2. **Compute is small.** You get $20 of HF credits, plus free Colab and Kaggle GPUs.
   GRPO with tool rollouts is slow: several samples per prompt, several turns per sample.
   Kaggle gives each account a weekly GPU quota. Three accounts give you three quotas.
3. **The deliverables are different.** No video is required. The required items are:
   a public GitHub repo, a write-up (HF Article or README), more than one before/after metric,
   and two public, un-gated HF collections (models/adapters, and datasets/logs/traces).
   An ROI comparison against a closed-source model is "optional but preferred". A HF Space is "if relevant".
   Judging criteria and weights are not published. The write-up requirements are the best guide.

## What judges will push back on

| Pushback | Why it hurts | Answer |
|---|---|---|
| "A FULL OUTER JOIN solves this. Why train an agent?" | With clean keys and rule-made anomalies, a 50-line script scores 100%. | Make the hard part something a fixed script cannot do: unknown schemas, no shared ID, references inside free-text fields, one deposit that settles many payments. Show the script as a baseline that fails on new schemas. |
| "It is synthetic. Does it work on real data?" | The model can learn the injector, not reconciliation. | Hold out one domain and one anomaly type. Add 10 to 20 hand-made episodes written by the teammate who did not write the injector. |
| "Where is the improvement?" | If Tiny Aya scores near 0 before and near 0 after, there is no project. | Warm-start with teacher trajectories, then RL. See the training plan below. |
| "Is the rounding trap a reward-hacking defence?" | No. It is a hard negative. A PS9 judge will say so. | Call it a hard negative. To claim PS9, run one ablation: recall-only reward, show the model flags everything, then show F1 reward fixes it. |
| "Is a timezone-shifted date an anomaly or noise?" | Ambiguous labels break exact ground truth. | Write one rule per anomaly type in the task prompt. Recommended: same instant in another timezone is noise. A real date difference is an anomaly. |
| "How did you use the toolkits?" | AuditKit docs conflict on agent metrics. Some trajectory metrics were deleted. | Use what is verified to work: custom `@ak.scorer` functions, `precomputed` outputs, `compare_models` with bootstrap. Check `failed_count` each run. |

## Other weak points

- **Scope.** Three systems, a paginated API, six anomaly types, random schemas, and a held-out domain is too much for 4.5 days.
- **Context length.** A small model cannot read thousands of rows. Raw `SELECT *` output will fill the context.
- **Cold-start RL.** GRPO needs reward variance inside each group. If all samples score 0, the model learns nothing.
- **Toolkit maturity.** AgentTune has 5 commits on main. It drops misspelled kwargs with no error. Its CLI has no `--tools` flag. Expect to lose hours here.
- **Agent build speed is not the bottleneck.** Fable, Opus, Sonnet and Haiku can write the environment in one day. GPU wall-clock time and toolkit bugs are the bottleneck. Agents cannot parallelise those.

## Revised plan

### Scope cuts for v1
- Two systems, one SQLite database, two tables. Add the third system only if v1 works.
- No paginated API in v1. Load the "API" data as a table.
- Two tools: `run_sql(query)` with a row cap (for example 20 rows) and `submit_findings(json)`.
- Small episodes: 20 to 60 rows per table, 3 to 6 planted items, 1 to 2 noise items.
- Four anomaly types in v1: missing, duplicate, amount mismatch, refund with no sale. One noise type: rounding.
- Difficulty levels: L1 shared ID and fixed schema. L2 random column names. L3 no shared ID, reference inside a text field.

### Training plan (the main change)
1. **Baselines:** Tiny Aya zero-shot, Aya Expanse 8B zero-shot, a strong teacher model, and the fixed SQL script.
2. **Teacher trajectories:** run a strong model on the environment. Keep only episodes with F1 = 1. Save them as AgentTune EventLogs.
3. **Warm start:** distill or SFT Tiny Aya with LoRA on those trajectories (`create_distill_trainer`). The docs say students should be under 3B parameters. Check that Tiny Aya is accepted.
4. **RL:** GRPO on top of the warm-started adapter (`create_agentic_trainer`, `algorithm="grpo"`, `tools=[...]`, custom reward).
5. This gives three bars for the write-up: zero-shot, after distillation, after RL. It covers PS6, PS3 and part of PS9.

### Reward
- F1 over (record id, anomaly type), with a small bonus for the correct amount.
- Penalty for each flagged noise item. Cap the number of findings so "flag everything" loses.
- Small penalty for failed SQL and for steps over a budget. Zero reward with no `submit_findings` call.
- Reward function signature: `f(completions: list[list[dict]], **kwargs) -> list[float]`.

### Evaluation (AuditKit)
- Metrics: precision, recall, F1, noise false-positive rate, valid-SQL rate, steps per episode, cost per episode.
- Splits: in-distribution, held-out domain, held-out anomaly type, hand-made set.
- `compare_models` with paired bootstrap for before/after. Report `failed_count`.
- ROI table: Tiny Aya on a rented GPU versus a closed-source model per 1,000 reconciliations.

### Schedule (IST)
- **Nov 10 night:** smoke test. Run the AgentTune GRPO quickstart with `CohereLabs/tiny-aya-global` on Kaggle. Accept the model licence first. One person owns this. In parallel, agents build the environment, injector and scorer.
- **Nov 11:** all four baselines. Start teacher trajectories. Decision gate: if GRPO does not run on Tiny Aya, the project is distillation only.
- **Nov 12:** distillation run. First GRPO run overnight.
- **Nov 13:** AuditKit evaluation on all splits. Reward ablation if GPU time remains.
- **Nov 14:** write-up, two HF collections, ROI table, README with reproduction steps, HF Space if time allows.
- **Nov 15:** buffer only. Submit by 12:00 noon IST, not 5:29 PM.

## Verification

- Injector unit tests: each planted item appears in the answer key. A perfect submission scores 1.0. An empty one scores 0. "Flag everything" scores below the zero-shot baseline.
- The fixed SQL script scores 1.0 on L1 and fails on L3. This proves the levels differ.
- The teacher model scores well above Tiny Aya zero-shot. If not, the task or prompt is broken.
- A 20-step GRPO run shows non-zero reward variance in the logs before any long run starts.
- All HF repos open in a private browser window with no login.

## Open risks

- Tiny Aya may not follow the tool format at all. Fallback: Aya Expanse 8B with 4-bit LoRA, if it fits the GPU.
- The Tiny Aya parameter count and Kaggle quota numbers are from my memory. Check both.
- AuditKit and AgentTune licences are noncommercial. That is fine for the hackathon.
