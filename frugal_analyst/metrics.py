"""Pure-Python metric helpers for trace records. Standard library only.

The evaluation script (scripts/eval_auditkit.py) and the demo app both use this file.
The script scores every episode through AuditKit. This file holds the same
per-episode formulas, a direct re-computation (`aggregate`), a strict trace
validator and a small paired bootstrap, so numbers can be cross-checked.

Metric directions: "max" means higher is better, "min" means lower is better.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Optional

SPLITS = ("train", "val", "test_id", "test_ood")
ANSWER_TYPES = ("number", "text")
COST_KEYS = ("tool_calls", "sql_calls", "failed_calls", "rows_returned", "vm_steps")


# ---------------------------------------------------------------------------
# Per-episode formulas. Each takes one trace dict and returns a float, or None
# when the value does not exist for that episode (for example latency_s is null).
# ---------------------------------------------------------------------------

def _num(x: Any) -> Optional[float]:
    return None if x is None else float(x)


def _call_ratio(t: dict) -> float:
    # Optimal tool calls is at least 1 (validated), so there is no zero division.
    return t["cost"]["tool_calls"] / t["optimal"]["tool_calls"]


def _vm_ratio(t: dict) -> float:
    # The +1 matches the efficiency formula in docs/spec.md and avoids zero division.
    return (t["cost"]["vm_steps"] + 1) / (t["optimal"]["vm_steps"] + 1)


def _efficiency_when_correct(t: dict) -> Optional[float]:
    # Only correct episodes count, so an accuracy change cannot hide inside this number.
    return float(t["efficiency"]) if t["correct"] else None


def _answer_in_sql_literal(t: dict) -> Optional[float]:
    # Old trace files do not have this field. None leaves them out instead of guessing 0.
    value = t.get("answer_in_sql_literal")
    return None if value is None else float(value)


# key -> (direction, label, per-episode function)
EPISODE_METRICS: dict[str, tuple[str, str, Callable[[dict], Optional[float]]]] = {
    "accuracy": ("max", "accuracy", lambda t: float(t["correct"])),
    "answered_rate": ("max", "answered rate", lambda t: float(t["answered"])),
    "grounded_rate": ("max", "grounded rate", lambda t: float(t["grounded"])),
    # An answer given with no query that read a data table: a guess.
    "unread_answer_rate": ("min", "answered with no table read",
                           lambda t: float(bool(t["answered"]) and not t["grounded"])),
    # Suspicion flag from reward.find_support. Not part of the reward.
    "answer_in_sql_literal_rate": ("min", "answer is a literal in its supporting SQL", _answer_in_sql_literal),
    "no_failed_call_rate": ("max", "episodes with 0 failed calls", lambda t: float(t["cost"]["failed_calls"] == 0)),
    "reward": ("max", "mean reward", lambda t: float(t["reward"])),
    "efficiency": ("max", "mean efficiency", lambda t: float(t["efficiency"])),
    "efficiency_when_correct": ("max", "efficiency (correct only)", _efficiency_when_correct),
    "correct_and_optimal_rate": (
        "max", "correct and optimal",
        lambda t: float(bool(t["correct"]) and t["cost"]["tool_calls"] <= t["optimal"]["tool_calls"]),
    ),
    "tool_calls": ("min", "mean tool calls", lambda t: float(t["cost"]["tool_calls"])),
    "failed_calls": ("min", "mean failed calls", lambda t: float(t["cost"]["failed_calls"])),
    "rows_returned": ("min", "mean rows returned", lambda t: float(t["cost"]["rows_returned"])),
    "vm_steps": ("min", "mean VM steps", lambda t: float(t["cost"]["vm_steps"])),
    "call_ratio": ("min", "call ratio to optimal", _call_ratio),
    "vm_step_ratio": ("min", "VM-step ratio to optimal", _vm_ratio),
    "latency_s": ("min", "mean latency (s)", lambda t: _num(t["latency_s"])),
    "completion_tokens": ("min", "mean completion tokens", lambda t: _num(t["completion_tokens"])),
}

# Metrics that exist only for some episodes (the rest are defined for every episode).
CONDITIONAL_METRICS = ("efficiency_when_correct", "answer_in_sql_literal_rate", "latency_s", "completion_tokens")

# Derived from sums, not a mean of per-episode values.
DERIVED_METRICS: dict[str, tuple[str, str]] = {
    "tool_call_validity": ("max", "tool-call validity (1 - failed/all calls)"),
}

ALL_METRIC_KEYS = list(EPISODE_METRICS) + list(DERIVED_METRICS)


def metric_direction(key: str) -> str:
    if key in EPISODE_METRICS:
        return EPISODE_METRICS[key][0]
    return DERIVED_METRICS[key][0]


def metric_label(key: str) -> str:
    if key in EPISODE_METRICS:
        return EPISODE_METRICS[key][1]
    return DERIVED_METRICS[key][1]


def episode_value(trace: dict, key: str) -> Optional[float]:
    """One episode's value for one metric key (None if it does not exist)."""
    return EPISODE_METRICS[key][2](trace)


def tool_call_validity(mean_failed_calls: float, mean_tool_calls: float) -> Optional[float]:
    """1 - failed/all calls over all episodes. Means have the same n, so the ratio of means is the ratio of sums."""
    if mean_tool_calls <= 0:
        return None
    return 1.0 - mean_failed_calls / mean_tool_calls


def mean(values: list[float]) -> Optional[float]:
    return math.fsum(values) / len(values) if values else None


def aggregate(traces: list[dict]) -> dict:
    """Direct pure-Python metrics for one model on one split.

    Returns {"n": int, "metrics": {key: float | None}, "counts": {key: n_used}}.
    Every episode is in every denominator. Only the conditional metrics
    (see CONDITIONAL_METRICS) use fewer episodes, and `counts` says how many.
    """
    metrics: dict[str, Optional[float]] = {}
    counts: dict[str, int] = {}
    for key in EPISODE_METRICS:
        values = [v for v in (episode_value(t, key) for t in traces) if v is not None]
        metrics[key] = mean(values)
        counts[key] = len(values)
    if metrics["tool_calls"] is not None and metrics["failed_calls"] is not None:
        metrics["tool_call_validity"] = tool_call_validity(metrics["failed_calls"], metrics["tool_calls"])
    else:
        metrics["tool_call_validity"] = None
    counts["tool_call_validity"] = len(traces)
    return {"n": len(traces), "metrics": metrics, "counts": counts}


def breakdown(traces: list[dict], field: str, keys: tuple[str, ...] = ("accuracy", "reward", "tool_calls")) -> dict:
    """Group traces by `field` ("level" or "domain") and average a few metrics per group."""
    groups: dict[Any, list[dict]] = {}
    for t in traces:
        groups.setdefault(t[field], []).append(t)
    out = {}
    for value in sorted(groups):
        agg = aggregate(groups[value])
        out[str(value)] = {"n": agg["n"], **{k: agg["metrics"][k] for k in keys}}
    return out


# ---------------------------------------------------------------------------
# Strict validation of one trace record against the schema in docs/spec.md.
# ---------------------------------------------------------------------------

def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_trace(rec: Any) -> list[str]:
    """Return a list of problems with one trace record. An empty list means the record is valid."""
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]
    errs: list[str] = []

    def need(key: str, check: Callable[[Any], bool], what: str) -> bool:
        if key not in rec:
            errs.append(f"missing key {key!r}")
            return False
        if not check(rec[key]):
            errs.append(f"{key!r} must be {what}, got {rec[key]!r}"[:200])
            return False
        return True

    is_str = lambda x: isinstance(x, str)  # noqa: E731
    is_bool = lambda x: isinstance(x, bool)  # noqa: E731
    need("task_id", lambda x: is_str(x) and x != "", "a non-empty string")
    need("domain", is_str, "a string")
    need("level", lambda x: _is_int(x) and x in (1, 2, 3), "1, 2 or 3")
    need("split", lambda x: x in SPLITS, f"one of {SPLITS}")
    need("question", is_str, "a string")
    need("gold_answer", is_str, "a string")
    need("answer_type", lambda x: x in ANSWER_TYPES, f"one of {ANSWER_TYPES}")
    need("model", lambda x: is_str(x) and x != "", "a non-empty string")
    need("predicted", lambda x: x is None or is_str(x), "a string or null")
    for key in ("answered", "grounded", "correct"):
        need(key, is_bool, "true or false")
    # Added with the evidence rule. Old trace files do not have them, so they are optional.
    for key in ("supported", "answer_in_sql_literal"):
        if key in rec and not is_bool(rec[key]):
            errs.append(f"{key!r} must be true or false, got {rec[key]!r}"[:200])
    need("efficiency", lambda x: _is_num(x) and 0.0 <= x <= 1.0, "a number in 0..1")
    need("reward", _is_num, "a number")
    need("raw_completion", lambda x: x is None or is_str(x), "a string or null")
    need("completion_tokens", lambda x: x is None or (_is_int(x) and x >= 0), "a non-negative integer or null")
    need("latency_s", lambda x: x is None or (_is_num(x) and x >= 0), "a non-negative number or null")

    n_tool_steps = None
    if need("steps", lambda x: isinstance(x, list), "a list"):
        n_tool_steps = 0
        for i, step in enumerate(rec["steps"]):
            if not isinstance(step, dict):
                errs.append(f"steps[{i}] is not an object")
            elif step.get("type") == "tool_call":
                n_tool_steps += 1
                for key, check, what in (
                    ("name", is_str, "a string"), ("arguments", lambda x: isinstance(x, dict), "an object"),
                    ("ok", is_bool, "true or false"), ("n_rows", lambda x: _is_int(x) and x >= 0, "a non-negative integer"),
                    ("vm_steps", lambda x: _is_int(x) and x >= 0, "a non-negative integer"), ("output", is_str, "a string"),
                ):
                    if key not in step:
                        errs.append(f"steps[{i}] missing key {key!r}")
                    elif not check(step[key]):
                        errs.append(f"steps[{i}].{key} must be {what}")
            elif step.get("type") == "final":
                if not is_str(step.get("text")):
                    errs.append(f"steps[{i}] final step needs a string 'text'")
            else:
                errs.append(f"steps[{i}].type must be 'tool_call' or 'final', got {step.get('type')!r}")

    cost = rec.get("cost")
    cost_ok = False
    if not isinstance(cost, dict):
        errs.append("'cost' must be an object")
    else:
        cost_ok = True
        for key in COST_KEYS:
            if key not in cost:
                errs.append(f"cost missing key {key!r}")
                cost_ok = False
            elif not (_is_int(cost[key]) and cost[key] >= 0):
                errs.append(f"cost.{key} must be a non-negative integer")
                cost_ok = False
    optimal = rec.get("optimal")
    optimal_ok = False
    if not isinstance(optimal, dict):
        errs.append("'optimal' must be an object")
    else:
        optimal_ok = True
        for key, minimum in (("tool_calls", 1), ("rows", 0), ("vm_steps", 0)):
            if key not in optimal:
                errs.append(f"optimal missing key {key!r}")
                optimal_ok = False
            elif not (_is_int(optimal[key]) and optimal[key] >= minimum):
                errs.append(f"optimal.{key} must be an integer >= {minimum}")
                optimal_ok = False

    # Cross-field checks. They need the fields above to be well formed.
    if cost_ok:
        if cost["failed_calls"] > cost["tool_calls"] or cost["sql_calls"] > cost["tool_calls"]:
            errs.append("cost.failed_calls and cost.sql_calls cannot exceed cost.tool_calls")
        if n_tool_steps is not None and cost["tool_calls"] != n_tool_steps:
            errs.append(f"cost.tool_calls is {cost['tool_calls']} but steps has {n_tool_steps} tool_call steps")
    if not errs and rec["correct"] and not (rec["answered"] and rec["grounded"]):
        errs.append("correct is true but answered or grounded is false")
    if not errs and "supported" in rec:
        if rec["correct"] and not rec["supported"]:
            errs.append("correct is true but supported is false")
        if rec["supported"] and not (rec["answered"] and rec["grounded"]):
            errs.append("supported is true but answered or grounded is false")
        if rec.get("answer_in_sql_literal") and not rec["supported"]:
            errs.append("answer_in_sql_literal is true but supported is false")
    return errs


# ---------------------------------------------------------------------------
# Paired bootstrap. This is OUR helper, not an AuditKit function. AuditKit 1.1.11
# only gives a p-value (auditkit._bootstrap.paired_bootstrap), no confidence interval.
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    baseline: list[float], candidate: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Percentile bootstrap CI of the mean paired difference (candidate - baseline).

    Both lists must be aligned by task: item i of each list is the same task.
    Resamples tasks with replacement, so the pairing is kept.
    Returns delta, ci_low, ci_high, excludes_zero, n, n_boot, seed, alpha.
    """
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have the same length (aligned by task)")
    n = len(baseline)
    if n == 0:
        raise ValueError("need at least one pair")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    diffs = [c - b for b, c in zip(baseline, candidate)]
    delta = math.fsum(diffs) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choices(diffs, k=n)
        means.append(math.fsum(sample) / n)
    means.sort()
    lo_index = int(math.floor((alpha / 2) * n_boot))
    hi_index = min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)
    ci_low, ci_high = means[lo_index], means[hi_index]
    return {
        "delta": delta, "ci_low": ci_low, "ci_high": ci_high,
        "excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "n": n, "n_boot": n_boot, "seed": seed, "alpha": alpha,
    }


def paired_from_traces(
    baseline: list[dict], candidate: list[dict], key: str, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired bootstrap on one metric, pairing traces by task_id. Only shared task ids are used."""
    base = {t["task_id"]: t for t in baseline}
    cand = {t["task_id"]: t for t in candidate}
    shared = sorted(base.keys() & cand.keys())
    pairs = [(episode_value(base[i], key), episode_value(cand[i], key)) for i in shared]
    pairs = [(b, c) for b, c in pairs if b is not None and c is not None]
    if not pairs:
        raise ValueError(f"no paired episodes with a value for {key!r}")
    return paired_bootstrap_ci([b for b, _ in pairs], [c for _, c in pairs], n_boot=n_boot, seed=seed)
