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
  `ak.AgentEvalRunner` are importable in 1.1.11. `scripts/eval_auditkit.py` does not use them, because our metrics
  (VM steps, rows, optimal-cost ratios) do not exist there. It uses custom scorers. The tool-call accuracy numbers
  come from `AgentEvalRunner` in `scripts/eval_agent.py`. See the section "Tool-call accuracy" below.
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

## Tool-call accuracy (`scripts/eval_agent.py`)

Checked in `.venv/lib/python3.12/site-packages/auditkit/agent_eval/` (`runner.py`, `types.py`, `importers.py`,
`outcome.py`, `report.py`) and `auditkit/metrics/agent.py`, `auditkit/trace.py`.

How it runs. Each (model, split) group is one `AgentEvalRunner().run(AgentEvalSpec(mode="recorded", ...))`.
One `AgentCase` per task. Its `reference_turns` are the oracle calls from `data/tasks/<split>.jsonl`, one call per turn.
The trace steps are never the reference. One episode per trace, from `episode_from_sample`. The outcome is
`CustomPredicate(fn, version="1")` over the `correct` flag of the trace, because `AnswerAssertion` fails on `ANSWER: 42`.
The cases have no `budgets`: a case over `max_tool_calls` is reported as budget exhausted and not scored,
and a wasteful model must still be scored. Per-episode values are in `row.diagnostics[score]["value"]`.
The raw AuditKit runs are saved in `outputs/eval/agent_runs/<model>_<split>.json`.

| Metric | Dir | AuditKit scorer and what it measures |
|---|---|---|
| name precision, recall, F1 (headline: name F1) | up | `tool_call_f1_name`. Multiset match on tool name only. Precision: share of the calls that match an oracle call. Recall: share of the oracle calls that were made. An extra `SELECT *` lowers precision. Stopping after one call lowers recall. |
| argument F1 | up | `tool_call_f1`. Same, but a call matches only if name and all arguments are equal as JSON values. |
| repeated call share | down | `redundant_tool_calls`. Share of calls that exactly repeat an earlier call. Skips an episode with no calls. |
| loop-free | up | `agent_loop_detection`. 1.0 if no loop. 0.0 if one call is made 3 or more times, or the call sequence returns to an earlier call (a to b to a). Skips an episode with no calls. |
| schema-valid calls | up | `tool_call_validity`. Share of calls that fit the tool schemas (tool exists, required arguments present, no unknown argument, right types). I pass the two real schemas, so it checks argument names too. It does not run the SQL, so a query that fails in SQLite still counts as valid. `no_failed_call_rate` in `results.md` covers failed SQL. Skips an episode with no calls. |
| task success | up | The verdict of the case outcome, from the `correct` flag. Cross-check of the trace field, not an independent judge. |

Not used: `trajectory_match` (turn order, as strict as argument F1), `parallel_tool_calls` (no parallel calls),
`tool_permission` and `tool_selection` (judge), `retrieval` (no contexts), `task_completion` (needs a judge model).

Strictness, read this before you quote a number. Argument F1 compares the SQL text byte for byte (as a JSON string).
A query that is correct but worded differently from the oracle query counts as a miss. A model that finds the right
answer in its own way scores low on argument F1 and high on task success. Name F1 does not look at the SQL, so it answers a
different question: did the model call the right tools, in the right number? It cannot tell a right query from a wrong one.
That is why name F1 is the headline (it does not punish equivalent SQL), argument F1 is reported next to it, and task
success is the check on the answer. `--headline tool_call_f1` switches the headline.

Directions that look odd. `naive` has name recall 1.0 and task success 1.0, but name precision about 0.5, since it
calls `get_schema` and runs `SELECT *` before the gold query. `zero-shot-stop` has name precision near 1.0 and recall
near 0.45, since it stops after the first call. Name F1 is one number for both. Read precision and recall too.

Cross-check (exit code 3). For each episode the script recomputes name-level and argument-level precision, recall and F1
in plain Python and compares with AuditKit. The difference limit is 1e-6. AuditKit uses a maximum matching on calls,
so duplicates count (multiset), unlike set-based F1: calling the same query twice adds a false positive. The plain
Python version counts `min(count in trace, count in reference)` per key, which is the same number, because for an exact
match "same name and arguments" is an equivalence relation. One known difference in definition: AuditKit compares
arguments as JSON values (`1` equals `1.0`), and the script compares `json.dumps` text. Our arguments are strings, so
the two agree here. If a trace ever had numeric arguments, the check could report a false mismatch.

Failed episodes (exit code 1). An episode is failed if its AuditKit status is not `completed` or its outcome is not
success or failure. An episode with no calls is not failed: its F1 scores are 0 and its repeat, loop and validity
scores are left out (the cell shows the count, for example `(80/120)`).

Paired comparison. Our `metrics.paired_bootstrap_ci` (AuditKit has no confidence interval), on the task ids both
models have. It runs for name F1 and argument F1. `ak.compare` is not used: it takes `RunResult`, and the agent
evaluator returns `AgentEvalResult`.

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
