"""Evaluate trace files with AuditKit.

    .venv/bin/python scripts/eval_auditkit.py --traces outputs/traces/*.jsonl --out outputs/eval/

The model outputs are already in the trace files, so no model runs here.
Each (model, split) group becomes one `ak.evaluate(..., model="precomputed")` call
with custom scorers. See docs/eval-notes.md for what was checked in AuditKit.

Exit codes: 0 ok, 1 AuditKit reported failed samples, 2 bad input, 3 AuditKit and the
pure-Python recomputation disagree.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python scripts/eval_auditkit.py` finds frugal_analyst

import auditkit as ak  # noqa: E402

from frugal_analyst import metrics as M  # noqa: E402
from frugal_analyst.prompts import build_user_prompt  # noqa: E402

EXIT_FAILED_SAMPLES, EXIT_BAD_INPUT, EXIT_MISMATCH = 1, 2, 3
MODEL_ORDER = ["zero-shot", "distilled", "grpo", "grpo-ungated", "naive", "oracle"]
COMPARED_METRICS = ["accuracy", "reward", "efficiency", "tool_calls"]
TOLERANCE = 1e-9  # AuditKit and the direct recomputation should agree to float rounding


class InputError(Exception):
    """A trace file is malformed or the inputs do not fit together."""


# ---------------------------------------------------------------------------
# Loading and validation. Nothing is skipped silently.
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace_file(path: Path) -> list[dict]:
    """Read one jsonl file. Raise InputError with file and line on the first bad record."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise InputError(f"{path}:{line_no}: invalid JSON ({e})") from None
            problems = M.validate_trace(rec)
            if problems:
                raise InputError(f"{path}:{line_no}: malformed trace record: " + "; ".join(problems))
            records.append(rec)
    if not records:
        raise InputError(f"{path}: file has no trace records")
    return records


def group_traces(records: list[tuple[str, dict]]) -> dict[str, dict[str, list[dict]]]:
    """Group by split, then by model label. A repeated task id inside one group is an error."""
    groups: dict[str, dict[str, list[dict]]] = {}
    seen: dict[tuple[str, str, str], str] = {}
    for path, rec in records:
        key = (rec["split"], rec["model"], rec["task_id"])
        if key in seen:
            raise InputError(
                f"duplicate task_id {rec['task_id']!r} for model {rec['model']!r} on split {rec['split']!r} "
                f"(in {seen[key]} and {path})")
        seen[key] = path
        groups.setdefault(rec["split"], {}).setdefault(rec["model"], []).append(rec)
    return groups


def coverage_report(models: dict[str, list[dict]]) -> dict:
    """Which task ids does each model cover, and which are missing for each model?"""
    ids = {label: {t["task_id"] for t in traces} for label, traces in models.items()}
    union = set().union(*ids.values())
    shared = set.intersection(*ids.values())
    return {
        "union": len(union),
        "shared": len(shared),
        "complete": union == shared,
        "missing": {label: sorted(union - got) for label, got in ids.items() if union - got},
    }


# ---------------------------------------------------------------------------
# AuditKit evaluation of one (model, split) group.
# ---------------------------------------------------------------------------

def make_scorer(key: str):
    """One AuditKit scorer for one metric. The per-episode formula lives in metrics.py."""
    direction = ak.Direction.MAXIMIZE if M.metric_direction(key) == "max" else ak.Direction.MINIMIZE

    def fn(sample, output):
        value = M.episode_value(sample.metadata["trace"], key)
        if value is None:
            # AuditKit's own convention for "nothing to measure": left out of the mean, not counted as a failure.
            return ak.Score(name=key, value=0.0, metadata={"unknown": True, "status": "not_tested"})
        return value

    return ak.scorer(fn, direction=direction, name=key)


SCORERS = [make_scorer(key) for key in M.EPISODE_METRICS]  # "accuracy" is first, so AuditKit's `correct` flag follows it


def make_samples(traces: list[dict]) -> list:
    samples = []
    for t in traces:
        compact = {k: t[k] for k in ("correct", "answered", "grounded", "efficiency", "reward", "cost", "optimal",
                                      "latency_s", "completion_tokens")}
        final = [s["text"] for s in t["steps"] if s["type"] == "final"]
        samples.append(ak.Sample(
            input=build_user_prompt(SimpleNamespace(domain=t["domain"], question=t["question"])),
            target=t["gold_answer"],
            id=t["task_id"],
            task=t["split"],
            actual_output=final[-1] if final else "",  # the model's final text. Empty for a truncated episode.
            metadata={"trace": compact, "level": t["level"], "domain": t["domain"], "model": t["model"]},
        ))
    return samples


def run_auditkit(traces: list[dict]):
    """Score precomputed outputs. cache=False so a result is never replayed from ~/.cache and nothing is written there."""
    return ak.evaluate(make_samples(traces), model="precomputed", scorers=SCORERS, cache=False)


def sample_values(run) -> dict[str, dict[str, float]]:
    """Per-episode values that AuditKit computed: {metric: {task_id: value}}. Left-out values are skipped."""
    out: dict[str, dict[str, float]] = {}
    for p in run.predictions:
        for doc in p.metadata.get("scores", []):
            meta = doc.get("metadata") or {}
            if meta.get("unknown"):
                continue
            out.setdefault(doc["name"], {})[p.sample_id] = float(doc["value"])
    return out


def auditkit_metrics(run, n_samples: int) -> tuple[dict, dict]:
    """Read the metric means and sample counts from the AuditKit result."""
    metrics, counts = {}, {}
    for key in M.EPISODE_METRICS:
        stat = run.stats.get(key)
        metrics[key] = stat.mean if stat is not None and stat.count else None
        counts[key] = stat.count if stat is not None else 0
    if metrics["tool_calls"] is not None and metrics["failed_calls"] is not None:
        metrics["tool_call_validity"] = M.tool_call_validity(metrics["failed_calls"], metrics["tool_calls"])
    else:
        metrics["tool_call_validity"] = None
    counts["tool_call_validity"] = n_samples
    return metrics, counts


def group_means(values: dict[str, dict[str, float]], traces: list[dict], field: str) -> dict:
    """Break AuditKit's per-episode values down by level or domain."""
    by_group: dict[str, list[str]] = {}
    for t in traces:
        by_group.setdefault(str(t[field]), []).append(t["task_id"])
    out = {}
    for group in sorted(by_group):
        row = {"n": len(by_group[group])}
        for key in ("accuracy", "reward", "tool_calls"):
            row[key] = M.mean([values[key][i] for i in by_group[group]])
        out[group] = row
    return out


def evaluate_group(label: str, split: str, traces: list[dict]) -> dict:
    """Run AuditKit on one group and cross-check it against the direct recomputation."""
    run = run_auditkit(traces)
    ak_metrics, ak_counts = auditkit_metrics(run, len(traces))
    direct = M.aggregate(traces)

    mismatches = []
    for key in M.ALL_METRIC_KEYS:
        a, d = ak_metrics[key], direct["metrics"][key]
        if (a is None) != (d is None) or (a is not None and abs(a - d) > TOLERANCE):
            mismatches.append({"metric": key, "auditkit": a, "direct": d})
        if ak_counts[key] != direct["counts"][key]:
            mismatches.append({"metric": key, "auditkit_n": ak_counts[key], "direct_n": direct["counts"][key]})
    # Also compare with fields stored in the traces, which do not use our formulas at all.
    stored = {
        "accuracy": M.mean([float(t["correct"]) for t in traces]),
        "reward": M.mean([float(t["reward"]) for t in traces]),
        "efficiency": M.mean([float(t["efficiency"]) for t in traces]),
    }
    for key, d in stored.items():
        if abs(ak_metrics[key] - d) > TOLERANCE:
            mismatches.append({"metric": key, "auditkit": ak_metrics[key], "stored_fields": d})

    values = sample_values(run)
    return {
        "run": run,
        "values": values,
        "result": {
            "n": len(traces),
            "metrics": ak_metrics,
            "counts": ak_counts,
            "auditkit_failed_samples": run.failed_count,
            "auditkit_errors": run.errors,
            "auditkit_unscored": run.unscored,
            "by_level": group_means(values, traces, "level"),
            "by_domain": group_means(values, traces, "domain"),
            "crosscheck": {"ok": not mismatches, "mismatches": mismatches},
        },
    }


# ---------------------------------------------------------------------------
# Paired comparison.
# ---------------------------------------------------------------------------

def verdict(delta: float, excludes_zero: bool, direction: str) -> str:
    if not excludes_zero:
        return "no clear change"
    good = delta > 0 if direction == "max" else delta < 0
    return "better" if good else "worse"


def compare_pair(base_label: str, cand_label: str, base_traces: list[dict], cand_traces: list[dict],
                 evaluated: dict, split: str, n_boot: int, seed: int) -> dict:
    """Baseline vs candidate on the task ids both have."""
    shared = {t["task_id"] for t in base_traces} & {t["task_id"] for t in cand_traces}
    shared_ids = sorted(shared)
    if not shared_ids:
        return {"model": cand_label, "n_paired": 0, "error": "no shared task ids"}

    def run_on_shared(label: str, traces: list[dict]) -> dict:
        # Reuse the full run when it already covers exactly the shared ids. Otherwise score the shared subset again.
        if len(traces) == len(shared_ids):
            return evaluated[(label, split)]
        return evaluate_group(label, split, [t for t in traces if t["task_id"] in shared])

    base_eval = run_on_shared(base_label, base_traces)
    cand_eval = run_on_shared(cand_label, cand_traces)
    comparison = ak.compare(base_eval["run"], cand_eval["run"])

    out = {"model": cand_label, "n_paired": len(shared_ids), "metrics": {}}
    for key in COMPARED_METRICS:
        base_vals = [base_eval["values"][key][i] for i in shared_ids]
        cand_vals = [cand_eval["values"][key][i] for i in shared_ids]
        ci = M.paired_bootstrap_ci(base_vals, cand_vals, n_boot=n_boot, seed=seed)
        direction = M.metric_direction(key)
        # AuditKit's own paired bootstrap gives a p-value only (no CI). Report it next to ours.
        ak_sig = comparison.significance(key)
        out["metrics"][key] = {
            "direction": direction,
            "baseline_mean": math.fsum(base_vals) / len(base_vals),
            "candidate_mean": math.fsum(cand_vals) / len(cand_vals),
            "delta": ci["delta"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "excludes_zero": ci["excludes_zero"],
            "verdict": verdict(ci["delta"], ci["excludes_zero"], direction),
            "auditkit_p_value": ak_sig.get("p_value"),
            "auditkit_delta": ak_sig.get("delta"),
            "auditkit_error": ak_sig.get("error"),
        }
    return out


# ---------------------------------------------------------------------------
# Reports.
# ---------------------------------------------------------------------------

def order_models(labels, baseline: str) -> list[str]:
    known = [m for m in MODEL_ORDER if m in labels]
    return sorted(labels, key=lambda m: (m != baseline, known.index(m) if m in known else 99, m))


def fmt(key: str, value, n: int | None = None, total: int | None = None) -> str:
    if value is None:
        return "n/a"
    if key == "vm_steps" or key == "rows_returned":
        text = f"{value:.0f}" if key == "vm_steps" else f"{value:.1f}"
    elif key in ("tool_calls", "failed_calls", "call_ratio", "vm_step_ratio", "latency_s", "completion_tokens"):
        text = f"{value:.2f}"
    else:
        text = f"{value:.3f}"
    if n is not None and total is not None and n != total:
        text += f" ({n}/{total})"
    return text


TABLE_KEYS = ["accuracy", "answered_rate", "grounded_rate", "tool_call_validity", "no_failed_call_rate", "reward",
              "efficiency", "efficiency_when_correct", "tool_calls", "rows_returned", "vm_steps", "call_ratio",
              "vm_step_ratio", "correct_and_optimal_rate", "latency_s", "completion_tokens"]


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def arrow(key: str) -> str:
    return "↑" if M.metric_direction(key) == "max" else "↓"


def render_markdown(results: dict) -> str:
    L = ["# Frugal Analyst evaluation results", ""]
    L.append(f"AuditKit {results['auditkit_version']}, seed {results['seed']}, bootstrap {results['bootstrap']} resamples. "
             f"Baseline: `{results['baseline']}`. All numbers are computed from the trace files by "
             f"`scripts/eval_auditkit.py`.")
    L.append(f"AuditKit failed samples (all runs): **{results['auditkit_failed_samples_total']}**. "
             f"AuditKit vs direct recomputation: **{'all agree' if results['crosscheck_ok'] else 'MISMATCH, see results.json'}**.")
    L += ["", "Arrows show the better direction: ↑ higher is better, ↓ lower is better. "
          "Failed, wrong and truncated episodes count as failures and stay in every denominator, "
          "except the three metrics marked below.", ""]
    L.append("Metrics with fewer episodes: `efficiency (correct only)` uses correct episodes only. "
             "`latency` and `completion tokens` use episodes where the trace has a value. "
             "A cell like `1.20 (80/120)` means 80 of 120 episodes. `n/a` means no episode had a value.")
    L.append("")
    for split, info in results["splits"].items():
        L.append(f"## Split `{split}`")
        L.append("")
        cov = info["coverage"]
        L.append(f"Task ids: {cov['union']} in total, {cov['shared']} shared by all models.")
        for label, missing in cov["missing"].items():
            L.append(f"- `{label}` is missing {len(missing)} task ids: {', '.join(missing[:20])}"
                     + (" ..." if len(missing) > 20 else ""))
        L.append("")
        header = ["model", "n"] + [f"{M.metric_label(k)} {arrow(k)}" for k in TABLE_KEYS]
        rows = []
        for label, m in info["models"].items():
            row = [f"`{label}`", str(m["n"])]
            for k in TABLE_KEYS:
                row.append(fmt(k, m["metrics"][k], m["counts"][k], m["n"] if k in M.CONDITIONAL_METRICS else None))
            rows.append(row)
        L += md_table(header, rows)
        L.append("")
        for field, title in (("by_level", "level"), ("by_domain", "domain")):
            L.append(f"Breakdown by {title}: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)")
            L.append("")
            groups = sorted({g for m in info["models"].values() for g in m[field]})
            rows = []
            for label, m in info["models"].items():
                row = [f"`{label}`"]
                for g in groups:
                    c = m[field].get(g)
                    row.append("n/a" if c is None else
                               f"{c['accuracy']:.3f} / {c['reward']:.3f} / {c['tool_calls']:.2f} (n={c['n']})")
                rows.append(row)
            L += md_table(["model"] + [f"{title} {g}" for g in groups], rows)
            L.append("")
    L.append("## Paired comparisons with the baseline")
    L.append("")
    L.append(f"Method: {results['comparison_method']}")
    L.append("")
    for split, comp in results["comparisons"].items():
        L.append(f"### Split `{split}`")
        L.append("")
        if comp.get("note"):
            L.append(comp["note"])
            L.append("")
        rows = []
        for pair in comp.get("pairs", []):
            if "error" in pair:
                rows.append([f"`{pair['model']}`", str(pair["n_paired"]), pair["error"], "", "", "", "", ""])
                continue
            for key in COMPARED_METRICS:
                c = pair["metrics"][key]
                p = c["auditkit_p_value"]
                rows.append([
                    f"`{pair['model']}`", str(pair["n_paired"]), f"{M.metric_label(key)} {arrow(key)}",
                    f"{c['baseline_mean']:.3f} to {c['candidate_mean']:.3f}", f"{c['delta']:+.3f}",
                    f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
                    "yes" if c["excludes_zero"] else "no", c["verdict"] + ("" if p is None else f" (AuditKit p={p:.3f})"),
                ])
        if rows:
            L += md_table(["vs baseline", "paired n", "metric", "baseline to model", "delta", "95% CI",
                           "CI excludes 0", "verdict"], rows)
            L.append("")
    L.append("## Inputs")
    L.append("")
    L += md_table(["file", "records", "sha256"], [[f"`{i['path']}`", str(i["records"]), f"`{i['sha256']}`"]
                                                   for i in results["inputs"]])
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matches:
            raise InputError(f"no files match {pattern!r}")
        paths += [Path(m) for m in matches]
    unique = list(dict.fromkeys(paths))
    for p in unique:
        if not p.is_file():
            raise InputError(f"not a file: {p}")
    return unique


def run(args: argparse.Namespace) -> int:
    if args.bootstrap < 1:
        raise InputError("--bootstrap must be at least 1")
    paths = expand_paths(args.traces)
    inputs, records = [], []
    for path in paths:
        recs = load_trace_file(path)
        inputs.append({"path": str(path), "records": len(recs), "sha256": sha256_of(path)})
        records += [(str(path), r) for r in recs]
    groups = group_traces(records)

    labels = {label for models in groups.values() for label in models}
    baseline = args.baseline or ("zero-shot" if "zero-shot" in labels else "naive")
    if baseline not in labels:
        raise InputError(f"baseline {baseline!r} is not among the model labels {sorted(labels)}")

    evaluated: dict[tuple[str, str], dict] = {}
    splits_out: dict[str, dict] = {}
    comparisons: dict[str, dict] = {}
    for split in sorted(groups):
        models = groups[split]
        cov = coverage_report(models)
        for label in order_models(models, baseline):
            evaluated[(label, split)] = evaluate_group(label, split, models[label])
        splits_out[split] = {
            "coverage": cov,
            "models": {label: evaluated[(label, split)]["result"] for label in order_models(models, baseline)},
        }
        comp: dict = {"baseline": baseline, "pairs": []}
        if baseline not in models:
            comp["note"] = f"Baseline `{baseline}` has no traces on this split, so there are no comparisons here."
        else:
            if not cov["complete"]:
                comp["note"] = ("Models do not cover the same task ids. Each pair is evaluated on the ids both "
                                "have, and the paired n shows the count.")
            for label in order_models(models, baseline):
                if label != baseline:
                    comp["pairs"].append(compare_pair(baseline, label, models[baseline], models[label],
                                                      evaluated, split, args.bootstrap, args.seed))
        comparisons[split] = comp

    failed_total = sum(e["result"]["auditkit_failed_samples"] for e in evaluated.values())
    crosscheck_ok = all(e["result"]["crosscheck"]["ok"] for e in evaluated.values())
    results = {
        "auditkit_version": ak.__version__,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "baseline": baseline,
        "inputs": inputs,
        "metric_info": {k: {"direction": M.metric_direction(k), "label": M.metric_label(k)} for k in M.ALL_METRIC_KEYS},
        "conditional_metrics": list(M.CONDITIONAL_METRICS),
        "splits": splits_out,
        "comparisons": comparisons,
        "comparison_method": (
            "Per-episode values come from AuditKit (ak.evaluate with model='precomputed' and custom scorers). "
            "Confidence intervals are a percentile paired bootstrap over tasks (frugal_analyst.metrics.paired_bootstrap_ci), "
            "because AuditKit 1.1.11 has no confidence interval. ak.compare(...).significance() adds AuditKit's own "
            "paired bootstrap p-value. ak.compare_models cannot take a different precomputed output per model, "
            "so it is not used."),
        "auditkit_failed_samples_total": failed_total,
        "crosscheck_ok": crosscheck_ok,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")

    print(f"AuditKit {ak.__version__}: {len(evaluated)} runs, failed samples total: {failed_total}")
    print(f"Wrote {out_dir / 'results.json'} and {out_dir / 'results.md'}")
    if failed_total:
        print("ERROR: AuditKit reported failed samples. See auditkit_errors in results.json.", file=sys.stderr)
        return EXIT_FAILED_SAMPLES
    if not crosscheck_ok:
        print("ERROR: AuditKit and the direct recomputation disagree. See crosscheck in results.json.", file=sys.stderr)
        return EXIT_MISMATCH
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="+", required=True, help="trace jsonl files (globs are expanded)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--baseline", default=None, help="baseline label. Default: zero-shot if present, else naive")
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
