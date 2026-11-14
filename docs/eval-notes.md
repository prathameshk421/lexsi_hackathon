# Evaluation notes: what was checked in AuditKit

Installed version: **auditkit 1.1.11** (`.venv/bin/python -c "import auditkit; print(auditkit.__version__)"`).
Where I looked: the installed source under `.venv/lib/python3.12/site-packages/auditkit/`
(`api.py`, `sample.py`, `scorers.py`, `score.py`, `report.py`, `runner.py`, `model/__init__.py`,
`model_compare.py`, `comparison.py`, `_bootstrap.py`, `runspec.py`, `cache.py`), plus small probe scripts that I ran.
I did not rely on the website docs. Only calls that I ran and saw work are used by the script.

## Calls the script uses (all run and working)

| Call | Real signature or fact |
|---|---|
| `ak.Sample` | dataclass. Used fields: `input`, `target`, `id`, `task`, `actual_output`, `metadata` (dict, free form). |
| `ak.scorer` | `ak.scorer(fn, *, direction, name=None)`. `direction` is required (`ak.Direction.MAXIMIZE` or `MINIMIZE`). `fn(sample, output)` returns a float or an `ak.Score`. |
| `ak.Score` | `ak.Score(name, value, ..., metadata={})`. With `metadata={"unknown": True, "status": "not_tested"}` the value is left out of the mean and is not a failure. |
| `ak.evaluate` | `ak.evaluate(dataset, model, scorers=None, *, ..., cache=True, **opts) -> RunResult`. `model="precomputed"` reads `Sample.actual_output` and calls no model. I pass `cache=False`. |
| `RunResult` | `.headline` (name to mean), `.stats[name]` (`.mean`, `.count`), `.predictions` (per sample, `.metadata["scores"]` holds each scorer value), `.failed_count`, `.errors`, `.unscored`, `.summary()`. |
| `ak.compare` | `ak.compare(baseline_run, candidate_run)` returns a `RunComparison`. `.significance(metric)` returns `{n, mean_baseline, mean_candidate, delta, p_value, significant}`. Delta is candidate minus baseline. |

## How precomputed outputs work

Each (model, split) group becomes one `ak.evaluate(samples, model="precomputed", scorers=[...], cache=False)` call.
`actual_output` is the final text of the episode (empty for a truncated episode). The numbers the scorers need
(cost, optimal cost, flags, latency) travel in `Sample.metadata["trace"]`. One scorer per metric, named by the metric key.
`Prediction.correct` in AuditKit follows the first scorer, so `accuracy` is first in the list.

## What did not work or does not exist

- **`ak.compare_models` cannot take a different precomputed output per model.** Its `models` argument is a list of
  specs, and `"precomputed"` reads the same `Sample.actual_output` for every model. I ran
  `compare_models(["precomputed", "precomputed"], samples, ...)` and both models got identical scores.
  A callable model that looks outputs up by prompt could work, but it is a workaround, so I did not use it.
  I use one `ak.evaluate` per model and `ak.compare` for the pair.
- **AuditKit has no confidence interval.** `auditkit._bootstrap.paired_bootstrap` (behind `.significance()`) returns a
  p-value from a bootstrap with the differences centred on zero. There is no CI anywhere. So the CI comes from our
  own `frugal_analyst.metrics.paired_bootstrap_ci`: a percentile bootstrap of the mean paired difference,
  resampling tasks with replacement, fixed seed. Its inputs are AuditKit's per-episode values. The script prints
  both our CI and AuditKit's p-value, and labels which is which.
- **Failed samples:** a scorer that raises is caught per sample. `RunResult.failed_count` goes up, the sample is
  left out of that scorer's mean, and a line goes into `.errors`. I tested this. The script exits with code 1 if the
  total is not 0. `RunConfig(failed_as_zero=True)` would count failures as 0.0, but we do not need it because any failure aborts.
- **Left-out values:** a `not_tested` score is counted in `.unscored`, not in `failed_count` and not in `.errors`. The script uses it for
  null latency, null completion tokens and for incorrect episodes in `efficiency_when_correct`.
- **Agent metrics:** `ak.ToolCallValidity`, `ak.TaskCompletion`, `ak.TrajectoryMatch`, `ak.RedundantToolCalls` and
  `ak.AgentEvalRunner` are importable in 1.1.11. I did not use them, because they expect AuditKit's own
  tool-call trace format, and our metrics (VM steps, rows, optimal-cost ratios) do not exist there. The notes from
  earlier that some agent metrics were removed were not checked further. Custom scorers are what the script uses.
- **Cache:** I set `cache=False`. In a probe, two runs with different `actual_output` did give different results with
  the cache on, so I saw no stale result. I still turn it off so nothing is replayed or written to `~/.cache/auditkit`.

## Metrics, directions and definitions

All per-episode formulas are in `frugal_analyst/metrics.py`. Arrows: up is better, down is better.

| Key | Dir | Definition |
|---|---|---|
| accuracy | up | `correct` |
| answered_rate, grounded_rate | up | `answered`, `grounded`. Since the evidence rule, `grounded` means a successful query read a data table. |
| unread_answer_rate | down | `answered` and not `grounded`: an answer with no query that read a data table. Defined for old and new traces. |
| answer_in_sql_literal_rate | down | the `answer_in_sql_literal` flag (see below). Old traces have no flag, so they are left out and the count is shown. |
| tool_call_validity | up | `1 - sum(failed_calls) / sum(tool_calls)` over all episodes. Derived from AuditKit's means (same n, so ratio of means = ratio of sums). `n/a` if no episode made a call. |
| no_failed_call_rate | up | share of episodes with `failed_calls == 0`. An episode with no calls counts as 0 failed. |
| reward, efficiency | up | stored `reward`, `efficiency` |
| efficiency_when_correct | up | `efficiency` over correct episodes only (fewer episodes, count shown) |
| tool_calls, rows_returned, vm_steps | down | `cost.*` |
| call_ratio | down | `cost.tool_calls / optimal.tool_calls`. An episode with no calls gives 0, so read it with accuracy. |
| vm_step_ratio | down | `(cost.vm_steps + 1) / (optimal.vm_steps + 1)`. The +1 is the same as in the efficiency formula in the spec. |
| correct_and_optimal_rate | up | `correct` and `cost.tool_calls <= optimal.tool_calls` |
| latency_s, completion_tokens | down | mean over episodes where the value is not null. The count is shown. |

Breakdowns by level and by domain: accuracy, reward and tool calls, from AuditKit's per-episode values.
No episode is excluded from any denominator, except for the four conditional metrics above.

There is no "supported and correct" rate. `correct` implies `supported`, so that rate equals accuracy.

## Evidence rule (reward change)

The old gated reward gave 1.0 to `SELECT 1` then the gold answer, to `SELECT '<gold>'`, and to
`SELECT '<gold>' FROM t LIMIT 1`. Now an answer is correct only if a query that read a data table returned it as a
cell, and the query did not type it into its top-level select list. `docs/spec.md` has the full rule.

`scripts/reward_audit.py` gives the evidence on the val split (40 tasks). Gated mean reward: oracle 1.0, oracle plus one
`LIMIT 5` preview 0.8442, naive 0.6431, `SELECT 1` then gold 0.05, `SELECT '<gold>'` 0.05,
`SELECT '<gold>' FROM t LIMIT 1` 0.1, gold with no query 0.0, honest wrong attempt 0.1.
A literal in any select list is caught at any depth: CTE, `FROM` subquery, `VALUES`, scalar subquery and `CASE`
result each score 0.1. The full table is in `outputs/eval/reward_audit.md`.

False positives: 0 of 660 oracle paths and 0 of 660 naive paths lose `supported` (all four splits). The test splits
were checked once as a sanity check of the rule. Nothing was tuned on them.

Known gaps, still open:
- An answer computed from other literals passes. Audit row "answer built by arithmetic or concatenation":
  `SELECT (360.0 / 2)` for gold 180, or `'Gar' || 'den'`, from a real table scores 0.9550 gated (correct on 38 of 40 val tasks;
  the other 2 fail only because the divisor `2` equals the gold answer `2`). The flag does not see this case.
- A filter value echoed back through a column passes (`SELECT grade_level FROM students WHERE grade_level = 10`).
  It looks the same as the 2 honest train tasks whose answer equals a `WHERE` value. The `answer_in_sql_literal` flag
  fires on it. The flag also fires on those 2 of 400 honest train oracle paths and on 0 val or test oracle paths.
  So read the flag rate as a warning sign during GRPO and evaluation, not as proof of a cheat.

`scripts/rescore_traces.py` replayed the 880 traces in `outputs/traces/` (8 files). Output text, ok flags and row counts
all match. 16 calls in 13 traces differ in VM steps only, so the run used `--allow-vm-drift`. One result changed:
`distilled` on `test_id`, task `retail-L3-0041`. Its only query used a guessed status code, returned no rows, and the
model still answered the gold category. The old reward gave 1.0, the new reward gives 0.1. `distilled` accuracy on
`test_id` goes from 1.000 to 0.992 (119 of 120). No other `correct`, `grounded` or `reward` value changed.

## Checks the script runs

1. Strict validation of every record (`metrics.validate_trace`). Errors name file and line. Nothing is skipped.
   Beyond the field types, it checks `cost.tool_calls` equals the number of `tool_call` steps, and `correct` implies `answered` and `grounded`.
   For traces that have `supported`: `correct` implies `supported`, `supported` implies `answered` and `grounded`,
   and `answer_in_sql_literal` implies `supported`. Old traces without these fields stay valid.
2. Duplicate `task_id` inside one (model, split) group is an error.
3. Coverage: ids missing per model are listed. Paired comparisons use the shared ids only, and `paired n` shows the count.
   The per-model tables still use all of that model's episodes.
4. Cross-check: AuditKit's means and counts against `metrics.aggregate` (direct recomputation), and accuracy, reward and
   efficiency against the stored trace fields. A difference above 1e-9 gives exit code 3.
   Limit: both paths share the per-episode formulas in `metrics.py`. The check proves that the AuditKit plumbing
   (sample alignment, scoring, aggregation) loses nothing. The formulas themselves are checked by hand-computed tests in `tests/test_metrics.py`.

Exit codes: 0 ok, 1 AuditKit failed samples, 2 bad input, 3 cross-check mismatch.
