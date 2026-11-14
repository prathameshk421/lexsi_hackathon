"""Score tool-call accuracy of trace files with AuditKit's agent evaluator.

    .venv/bin/python scripts/eval_agent.py --traces outputs/traces/*.jsonl --out outputs/eval/

The model outputs are already in the trace files, so no model runs here.
For each (model, split) group the script builds one AuditKit `AgentCase` per task.
The reference tool calls are the oracle calls from data/tasks/<split>.jsonl,
never the steps of the trace itself. Then `AgentEvalRunner` scores the recorded
episodes. See docs/eval-notes.md for what each scorer measures.

Writes agent_results.json, agent_results.md and one raw AuditKit run per group
into <out>/agent_runs/.

Exit codes: 0 ok, 1 AuditKit reported failed or errored episodes, 2 bad input,
3 AuditKit and the pure-Python recomputation disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python scripts/eval_agent.py` finds frugal_analyst
sys.path.insert(0, str(ROOT / "scripts"))  # and the sibling script that has the shared helpers

import auditkit as ak  # noqa: E402
from auditkit.agent_eval import (  # noqa: E402
    AgentCase, AgentEvalRunner, AgentEvalSpec, CustomPredicate, episode_from_sample,
)

import eval_auditkit as base  # noqa: E402
from eval_auditkit import InputError  # noqa: E402
from frugal_analyst import metrics as M  # noqa: E402

EXIT_FAILED_EPISODES, EXIT_BAD_INPUT, EXIT_MISMATCH = 1, 2, 3
TOLERANCE = 1e-6  # AuditKit and the direct recomputation should agree to float rounding
HEADLINE_CHOICES = ("tool_call_f1_name", "tool_call_f1")

# Scorer names that AuditKit's runner accepts (KNOWN_SCORERS in agent_eval/runner.py).
AUDITKIT_SCORERS = ["tool_call_f1", "tool_call_f1_name", "redundant_tool_calls", "agent_loop_detection",
                    "tool_call_validity"]

# key -> (label, direction, name of the AuditKit score that holds the per-episode value).
# `task_success` has no score name: it is the verdict of the case outcome.
METRICS = {
    "tool_call_precision_name": ("name precision", "max", "tool_call_precision_name"),
    "tool_call_recall_name": ("name recall", "max", "tool_call_recall_name"),
    "tool_call_f1_name": ("name F1", "max", "tool_call_f1_name"),
    "tool_call_f1": ("argument F1", "max", "tool_call_f1"),
    "redundant_tool_calls": ("repeated call share", "min", "redundant_tool_calls"),
    "agent_loop_detection": ("loop-free", "max", "agent_loop_detection"),
    "tool_call_validity": ("schema-valid calls", "max", "tool_call_validity"),
    "task_success": ("task success", "max", None),
}
# The F1 metrics are defined for every episode. The other three skip an episode with no calls.
ALWAYS_DEFINED = ("tool_call_precision_name", "tool_call_recall_name", "tool_call_f1_name", "tool_call_f1",
                  "task_success")

# The two tools as the model sees them (see TOOLS in frugal_analyst/env.py).
# AuditKit's validity scorer needs schemas to check argument names, not only tool names.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_schema", "description": "List the tables and columns of a database.",
        "parameters": {"type": "object", "properties": {"database": {"type": "string"}},
                       "required": ["database"]}}},
    {"type": "function", "function": {
        "name": "run_sql", "description": "Run one read-only SQL query.",
        "parameters": {"type": "object",
                       "properties": {"database": {"type": "string"}, "query": {"type": "string"}},
                       "required": ["database", "query"]}}},
]


# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------

def expand_trace_paths(patterns: list[str]) -> list[Path]:
    """Expand globs and drop the `<trace>.run.json` harness files, which are not traces."""
    paths = [p for p in base.expand_paths(patterns) if not p.name.endswith(".run.json")]
    if not paths:
        raise InputError("no trace files left after skipping .run.json files")
    return paths


def load_oracle_calls(tasks_dir: Path, split: str) -> dict[str, list[dict]]:
    """Oracle tool calls by task id for one split. A broken file is an error, not a skip."""
    path = tasks_dir / f"{split}.jsonl"
    if not path.is_file():
        raise InputError(f"no task file for split {split!r}: {path}")
    out: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
                calls = task["oracle_calls"]
                ok = (isinstance(calls, list) and calls
                      and all(isinstance(c, dict) and isinstance(c.get("name"), str)
                              and isinstance(c.get("arguments"), dict) for c in calls))
                task_id = task["id"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise InputError(f"{path}:{line_no}: bad task record ({e!r})") from None
            if not ok:
                raise InputError(f"{path}:{line_no}: oracle_calls must be a non-empty list of name and arguments")
            out[task_id] = [{"name": c["name"], "arguments": c["arguments"]} for c in calls]
    return out


# ---------------------------------------------------------------------------
# AuditKit objects.
# ---------------------------------------------------------------------------

def trace_calls(trace: dict) -> list[dict]:
    """The calls the model made, in order, as plain name and arguments dicts."""
    return [{"name": s["name"], "arguments": s["arguments"]} for s in trace["steps"] if s["type"] == "tool_call"]


def final_text(trace: dict) -> str:
    final = [s["text"] for s in trace["steps"] if s["type"] == "final"]
    return final[-1] if final else ""  # empty for a truncated episode


def is_correct(episode, case) -> bool:
    """Case outcome. It reads the `correct` flag of the trace, which our reward set.

    AuditKit's `AnswerAssertion` fails on our `ANSWER: 42` text, so a custom check is used.
    """
    return bool(case.metadata["correct"])


def make_case(trace: dict, oracle_calls: list[dict]) -> AgentCase:
    # One call per turn. The F1 scorers ignore turn borders. No `budgets`: AuditKit would then
    # label a model with more calls than optimal as "budget exhausted" instead of scoring it.
    return AgentCase(
        id=trace["task_id"], task=trace["question"],
        reference_turns=[[c] for c in oracle_calls],
        outcome=CustomPredicate(is_correct, version="1"),
        allowed_tools=TOOL_SCHEMAS,
        metadata={"correct": bool(trace["correct"])},
    )


def make_episode(trace: dict):
    sample = ak.Sample(
        input=trace["question"], target=trace["gold_answer"], id=trace["task_id"],
        actual_output=final_text(trace),
        actual_trace={"tool_calls": [[c] for c in trace_calls(trace)]},  # [] means "observed, no calls"
    )
    episode = episode_from_sample(sample)
    episode.case_id = trace["task_id"]
    return episode


def run_auditkit(label: str, split: str, traces: list[dict], oracle: dict[str, list[dict]]):
    missing = sorted(t["task_id"] for t in traces if t["task_id"] not in oracle)
    if missing:
        raise InputError(f"model {label!r}, split {split!r}: {len(missing)} task ids have no oracle calls "
                         f"in the task file, for example {missing[:3]}")
    cases = [make_case(t, oracle[t["task_id"]]) for t in traces]
    episodes = [make_episode(t) for t in traces]
    spec = AgentEvalSpec(cases=cases, episodes=episodes, mode="recorded", scorers=AUDITKIT_SCORERS,
                         name=f"{label}-{split}")
    return AgentEvalRunner().run(spec)


def episode_values(result) -> dict[str, dict[str, float]]:
    """Per-episode values that AuditKit computed: {metric: {task_id: value}}. Unscored episodes are absent."""
    out: dict[str, dict[str, float]] = {key: {} for key in METRICS}
    for row in result.rows:
        for key, (_, _, score_name) in METRICS.items():
            if score_name is None:
                verdict = row.outcome.get("verdict")
                if verdict in ("success", "failure"):
                    out[key][row.case_id] = 1.0 if verdict == "success" else 0.0
                continue
            doc = row.diagnostics.get(score_name)
            if doc is not None and doc.get("value") is not None:
                out[key][row.case_id] = float(doc["value"])
    return out


def failed_episodes(result) -> list[dict]:
    """Episodes AuditKit could not score: any status but `completed`, or an outcome that is not decided.

    An episode with no calls is not a failure. Its repeat, loop and validity scores are left out, and counted as such.
    """
    bad = []
    for row in result.rows:
        verdict = row.outcome.get("verdict")
        if row.status != "completed" or verdict not in ("success", "failure"):
            bad.append({"task_id": row.case_id, "status": row.status, "verdict": verdict,
                        "reason": row.outcome.get("reason"), "errors": row.errors,
                        "ineligible": row.ineligible})
    return bad


# ---------------------------------------------------------------------------
# Cross-check: the same numbers in plain Python.
# ---------------------------------------------------------------------------

def call_key(call: dict, with_args: bool):
    return (call["name"], json.dumps(call["arguments"], sort_keys=True)) if with_args else call["name"]


def multiset_prf(predicted: list[dict], reference: list[dict], with_args: bool) -> dict[str, float]:
    """Multiset precision, recall and F1, the way AuditKit's `ToolCallF1` defines them.

    A call matches one reference call with the same name (and the same arguments when `with_args`).
    Each reference call matches at most once, so a repeated call counts again as a false positive.
    A set-based F1 would ignore the repeat. For an exact match this equals AuditKit's maximum matching,
    because "same name and arguments" is an equivalence relation.
    """
    pred = Counter(call_key(c, with_args) for c in predicted)
    ref = Counter(call_key(c, with_args) for c in reference)
    matched = sum(min(n, ref[k]) for k, n in pred.items())
    if not reference:  # AuditKit: with an empty reference everything is 1.0 iff there are no calls
        p = r = f = 1.0 if not predicted else 0.0
    else:
        p = matched / len(predicted) if predicted else 0.0
        r = matched / len(reference)
        f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def crosscheck(traces: list[dict], oracle: dict[str, list[dict]], values: dict[str, dict[str, float]]) -> dict:
    """Compare AuditKit's per-episode values with the plain Python ones. A difference above 1e-6 is a mismatch."""
    mismatches = []
    checks = [("tool_call_precision_name", False, "precision"), ("tool_call_recall_name", False, "recall"),
              ("tool_call_f1_name", False, "f1"), ("tool_call_f1", True, "f1")]
    for t in traces:
        reference, predicted = oracle[t["task_id"]], trace_calls(t)
        for key, with_args, field in checks:
            direct = multiset_prf(predicted, reference, with_args)[field]
            got = values[key].get(t["task_id"])
            if got is None or abs(got - direct) > TOLERANCE:
                mismatches.append({"task_id": t["task_id"], "metric": key, "auditkit": got, "direct": direct})
    return {"ok": not mismatches, "checked_episodes": len(traces), "mismatches": mismatches[:20],
            "n_mismatches": len(mismatches)}


# ---------------------------------------------------------------------------
# One (model, split) group and the paired comparison.
# ---------------------------------------------------------------------------

def mean_or_none(vals: list[float]) -> float | None:
    return math.fsum(vals) / len(vals) if vals else None


def group_means(values: dict[str, dict[str, float]], traces: list[dict], headline: str) -> dict:
    """Break the headline, argument F1 and task success down by template group."""
    by_group: dict[str, list[str]] = {}
    for t in traces:
        by_group.setdefault(base.template_group(t), []).append(t["task_id"])
    out = {}
    for group in sorted(by_group):
        row = {"n": len(by_group[group])}
        for key in dict.fromkeys([headline, "tool_call_f1", "task_success"]):
            row[key] = mean_or_none([values[key][i] for i in by_group[group] if i in values[key]])
        out[group] = row
    return out


def evaluate_group(label: str, split: str, traces: list[dict], oracle: dict, headline: str, runs_dir: Path | None):
    result = run_auditkit(label, split, traces, oracle)
    values = episode_values(result)
    failed = failed_episodes(result)
    if runs_dir is not None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        result.to_json(str(runs_dir / f"{label}_{split}.json"))
    return {
        "values": values,
        "result": {
            "n": len(traces),
            "metrics": {k: mean_or_none(list(v.values())) for k, v in values.items()},
            "counts": {k: len(v) for k, v in values.items()},
            "failed_episodes": len(failed),
            "failures": failed[:20],
            "auditkit_summary": result.summary(),
            "by_template_group": group_means(values, traces, headline),
            "crosscheck": crosscheck(traces, oracle, values),
        },
    }


def compare_pair(base_label: str, cand_label: str, evaluated: dict, split: str, keys: list[str],
                 n_boot: int, seed: int) -> dict:
    """Baseline vs candidate on the task ids that both models have."""
    bv, cv = evaluated[(base_label, split)]["values"], evaluated[(cand_label, split)]["values"]
    out = {"model": cand_label, "metrics": {}}
    for key in keys:
        shared = sorted(bv[key].keys() & cv[key].keys())
        if not shared:
            out["metrics"][key] = {"n_paired": 0, "error": "no shared task ids with a value"}
            continue
        b, c = [bv[key][i] for i in shared], [cv[key][i] for i in shared]
        ci = M.paired_bootstrap_ci(b, c, n_boot=n_boot, seed=seed)
        direction = METRICS[key][1]
        out["metrics"][key] = {
            "n_paired": len(shared), "direction": direction,
            "baseline_mean": math.fsum(b) / len(b), "candidate_mean": math.fsum(c) / len(c),
            "delta": ci["delta"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "excludes_zero": ci["excludes_zero"],
            "verdict": base.verdict(ci["delta"], ci["excludes_zero"], direction),
        }
    return out


# ---------------------------------------------------------------------------
# Reports.
# ---------------------------------------------------------------------------

def arrow(key: str) -> str:
    return "↑" if METRICS[key][1] == "max" else "↓"


def fmt(value, n: int | None = None, total: int | None = None) -> str:
    if value is None:
        return "n/a"
    text = f"{value:.3f}"
    if n is not None and total is not None and n != total:
        text += f" ({n}/{total})"
    return text


def render_markdown(results: dict) -> str:
    headline = results["headline"]
    L = ["# Frugal Analyst tool-call accuracy (AuditKit agent evaluator)", ""]
    L.append(f"AuditKit {results['auditkit_version']}, seed {results['seed']}, bootstrap {results['bootstrap']} resamples. "
             f"Baseline: `{results['baseline']}`. Reference calls: the oracle calls of each task. "
             f"All numbers come from the trace files through `scripts/eval_agent.py`.")
    L.append(f"Headline metric: **{METRICS[headline][0]}** (`{headline}`). "
             f"Failed or errored episodes (all runs): **{results['failed_episodes_total']}**. "
             f"AuditKit vs direct recomputation: **{'all agree' if results['crosscheck_ok'] else 'MISMATCH, see agent_results.json'}**.")
    L += ["", "Arrows show the better direction: ↑ higher is better, ↓ lower is better. "
          "Name metrics compare tool names only. The argument F1 needs the same SQL text as the oracle, "
          "so a correct query with other wording counts as a miss there. "
          "A cell like `0.250 (80/120)` means 80 of 120 episodes had a value. "
          "Repeat, loop and validity scores skip an episode that made no call.", ""]
    keys = list(METRICS)
    for split, info in results["splits"].items():
        L += [f"## Split `{split}`", ""]
        cov = info["coverage"]
        L.append(f"Task ids: {cov['union']} in total, {cov['shared']} shared by all models.")
        for label, missing in cov["missing"].items():
            L.append(f"- `{label}` is missing {len(missing)} task ids: {', '.join(missing[:20])}"
                     + (" ..." if len(missing) > 20 else ""))
        L.append("")
        header = ["model", "n", "failed"] + [f"{METRICS[k][0]} {arrow(k)}" for k in keys]
        rows = []
        for label, m in info["models"].items():
            rows.append([f"`{label}`", str(m["n"]), str(m["failed_episodes"])]
                        + [fmt(m["metrics"][k], m["counts"][k], m["n"]) for k in keys])
        L += base.md_table(header, rows)
        L.append("")
        groups = sorted({g for m in info["models"].values() for g in m["by_template_group"]})
        if len(groups) > 1:
            L.append(f"Breakdown by template group (rl_only: never in the SFT data): "
                     f"{METRICS[headline][0]} ↑ / argument F1 ↑ / task success ↑ (n)")
            L.append("")
            rows = []
            for label, m in info["models"].items():
                row = [f"`{label}`"]
                for g in groups:
                    c = m["by_template_group"].get(g)
                    row.append("n/a" if c is None else
                               f"{fmt(c[headline])} / {fmt(c['tool_call_f1'])} / {fmt(c['task_success'])} (n={c['n']})")
                rows.append(row)
            L += base.md_table(["model"] + groups, rows)
            L.append("")
    L += ["## Paired comparisons with the baseline", "", f"Method: {results['comparison_method']}", ""]
    for split, comp in results["comparisons"].items():
        L += [f"### Split `{split}`", ""]
        if comp.get("note"):
            L += [comp["note"], ""]
        rows = []
        for pair in comp.get("pairs", []):
            for key, c in pair["metrics"].items():
                name = f"{METRICS[key][0]} {arrow(key)}" + (" (headline)" if key == headline else "")
                if "error" in c:
                    rows.append([f"`{pair['model']}`", name, "0", c["error"], "", "", "", ""])
                    continue
                rows.append([f"`{pair['model']}`", name, str(c["n_paired"]),
                             f"{c['baseline_mean']:.3f} to {c['candidate_mean']:.3f}", f"{c['delta']:+.3f}",
                             f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
                             "yes" if c["excludes_zero"] else "no", c["verdict"]])
        if rows:
            L += base.md_table(["vs baseline", "metric", "paired n", "baseline to model", "delta", "95% CI",
                                "CI excludes 0", "verdict"], rows)
            L.append("")
    L += ["## Inputs", ""]
    L += base.md_table(["file", "records", "sha256"], [[f"`{i['path']}`", str(i["records"]), f"`{i['sha256']}`"]
                                                       for i in results["inputs"]])
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    if args.bootstrap < 1:
        raise InputError("--bootstrap must be at least 1")
    inputs, records = [], []
    for path in expand_trace_paths(args.traces):
        recs = base.load_trace_file(path)
        inputs.append({"path": str(path), "records": len(recs), "sha256": base.sha256_of(path)})
        records += [(str(path), r) for r in recs]
    groups = base.group_traces(records)

    labels = {label for models in groups.values() for label in models}
    baseline = args.baseline or ("zero-shot" if "zero-shot" in labels else "naive")
    if baseline not in labels:
        raise InputError(f"baseline {baseline!r} is not among the model labels {sorted(labels)}")

    out_dir = Path(args.out)
    tasks_dir = Path(args.tasks_dir)
    headline = args.headline
    compared = list(dict.fromkeys([headline, "tool_call_f1_name", "tool_call_f1"]))

    evaluated: dict[tuple[str, str], dict] = {}
    splits_out: dict[str, dict] = {}
    comparisons: dict[str, dict] = {}
    for split in sorted(groups):
        models = groups[split]
        oracle = load_oracle_calls(tasks_dir, split)
        ordered = base.order_models(models, baseline)
        for label in ordered:
            evaluated[(label, split)] = evaluate_group(label, split, models[label], oracle, headline,
                                                       out_dir / "agent_runs")
        splits_out[split] = {"coverage": base.coverage_report(models),
                             "models": {label: evaluated[(label, split)]["result"] for label in ordered}}
        comp: dict = {"baseline": baseline, "pairs": []}
        if baseline not in models:
            comp["note"] = f"Baseline `{baseline}` has no traces on this split, so there are no comparisons here."
        else:
            if not splits_out[split]["coverage"]["complete"]:
                comp["note"] = "Models do not cover the same task ids. Each pair uses the ids both have."
            comp["pairs"] = [compare_pair(baseline, label, evaluated, split, compared, args.bootstrap, args.seed)
                             for label in ordered if label != baseline]
        comparisons[split] = comp

    failed_total = sum(e["result"]["failed_episodes"] for e in evaluated.values())
    crosscheck_ok = all(e["result"]["crosscheck"]["ok"] for e in evaluated.values())
    results = {
        "auditkit_version": ak.__version__,
        "seed": args.seed, "bootstrap": args.bootstrap, "baseline": baseline, "headline": headline,
        "tasks_dir": str(tasks_dir),
        "inputs": inputs,
        "scorers": AUDITKIT_SCORERS,
        "metric_info": {k: {"label": v[0], "direction": v[1], "auditkit_score": v[2]} for k, v in METRICS.items()},
        "splits": splits_out,
        "comparisons": comparisons,
        "comparison_method": (
            "Per-episode values come from AuditKit (AgentEvalRunner, recorded mode). "
            "Confidence intervals are a percentile paired bootstrap over tasks (frugal_analyst.metrics.paired_bootstrap_ci), "
            "because AuditKit 1.1.11 has no confidence interval."),
        "failed_episodes_total": failed_total,
        "crosscheck_ok": crosscheck_ok,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agent_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "agent_results.md").write_text(render_markdown(results), encoding="utf-8")

    print(f"AuditKit {ak.__version__}: {len(evaluated)} runs, failed episodes total: {failed_total}")
    print(f"Wrote {out_dir / 'agent_results.json'} and {out_dir / 'agent_results.md'}")
    if failed_total:
        print("ERROR: AuditKit reported failed or errored episodes. See failures in agent_results.json.", file=sys.stderr)
        return EXIT_FAILED_EPISODES
    if not crosscheck_ok:
        print("ERROR: AuditKit and the direct recomputation disagree. See crosscheck in agent_results.json.", file=sys.stderr)
        return EXIT_MISMATCH
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="+", required=True, help="trace jsonl files (globs are expanded)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--tasks-dir", default=str(ROOT / "data" / "tasks"), help="folder with <split>.jsonl task files")
    parser.add_argument("--baseline", default=None, help="baseline label. Default: zero-shot if present, else naive")
    parser.add_argument("--headline", default="tool_call_f1_name", choices=HEADLINE_CHOICES,
                        help="metric for the headline tool-call accuracy")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())
