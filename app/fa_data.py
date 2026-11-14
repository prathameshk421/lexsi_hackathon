"""Read trace files and task files for the demo.

Every file the app reads is chosen from a fixed list on the server. A name that
comes from the UI is only used as a key into what `discover` found on disk. It is
never joined into a path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO_ROOT / "outputs" / "traces"
TASK_DIR = REPO_ROOT / "data" / "tasks"
RESULTS_PATH = REPO_ROOT / "outputs" / "eval" / "results.json"
# Optional comparison files of the same shape, with another baseline. The Results tab reads them if they exist.
EXTRA_RESULTS = {
    "vs_sft": REPO_ROOT / "outputs" / "eval_vs_sft" / "results.json",
    "mixed": REPO_ROOT / "outputs" / "eval_mixed" / "results.json",
}

SPLITS = ("test_id", "test_ood", "val", "train")
EXAMPLE_SPLITS = ("val", "test_id", "test_ood")  # task files used for the Ask examples
SCRIPTED_LABELS = ("oracle", "naive", "guess")

# Every run name the app shows comes from this one table: label -> (display name, short name, one-line note).
# The order is the display order: baseline, SFT, GRPO, experiments, references. It is not a quality ranking.
# The notes follow the run table in the top-level README.md.
RUN_INFO = {
    "zero-shot": ("Zero-shot (base model)", "Zero-shot",
                  "Tiny Aya with the two tools and no training. The baseline for every headline number."),
    "distilled": ("SFT, 274 tasks", "SFT",
                  "LoRA SFT on scripted oracle traces of 274 train tasks. 12 templates are held back for RL."),
    "grpo": ("SFT + GRPO (final)", "SFT + GRPO",
             "The SFT model plus 50 GRPO steps with the evidence-gated cost reward. The final model."),
    "grpo-ungated": ("SFT + GRPO, ungated reward (ablation)", "GRPO ungated",
                     "Like the final model, but 30 GRPO steps with no evidence gate in the reward."),
    "distilled-mixed": ("SFT, mixed teacher", "SFT mixed",
                        "SFT on a teacher that adds one needless preview query to half of the traces, on purpose."),
    "grpo-mixed": ("SFT + GRPO, mixed teacher", "GRPO mixed",
                   "The mixed-teacher SFT model plus 30 GRPO steps. It tests if GRPO removes the copied waste."),
    "zero-shot-stop": ("Zero-shot, stop-after-call harness", "Zero-shot stop",
                       "The base model with each turn cut after a tool call. It often never answers. A negative result."),
    "distilled-all": ("SFT, all 400 tasks (reference)", "SFT all 400",
                      "SFT on all 400 train tasks, the first design. A reference, not part of the main pipeline."),
    "oracle": ("Oracle (scripted)", "Oracle",
               "A fixed script that replays the efficient gold path. Not a model. It sets the optimal cost."),
    "naive": ("Naive (scripted)", "Naive",
              "A fixed script that gets the right answer the wasteful way. Not a model."),
    "guess": ("Guess (scripted)", "Guess", "A fixed script that answers with no query. Not a model."),
}
LABEL_ORDER = list(RUN_INFO)
BASELINE_LABEL = "zero-shot"
# The run that the first screen compares with the baseline: the final model, else SFT.
PREFERRED_AFTER = ("grpo", "distilled")

MAX_FILE_BYTES = 80 * 1024 * 1024
_FILE_RE = re.compile(r"^([a-z0-9][a-z0-9-]{0,30})_(test_id|test_ood|val|train)\.jsonl$")

LEVEL_TEXT = {
    1: "Level 1: one table, a filter and an aggregate",
    2: "Level 2: a join and an aggregate or top-1",
    3: "Level 3: needs a coded value, so it needs a lookup first",
}


def label_sort_key(label: str) -> tuple[int, int, str]:
    """Models in RUN_INFO order, then unknown labels, scripted reference policies last."""
    scripted = 1 if label in SCRIPTED_LABELS else 0
    rank = LABEL_ORDER.index(label) if label in LABEL_ORDER else len(LABEL_ORDER)
    return (scripted, rank, label)


def display_label(label) -> str:
    """Plain name for a run label. An unknown label is shown as it is."""
    info = RUN_INFO.get(label)
    return info[0] if info else str(label)


def short_label(label) -> str:
    """A short name for narrow places such as the cost meters."""
    info = RUN_INFO.get(label)
    return info[1] if info else str(label)


def run_note(label) -> str:
    info = RUN_INFO.get(label)
    return info[2] if info else ""


def choice_label(label) -> str:
    """Dropdown text: the plain name, then the label that the write-up uses."""
    name = display_label(label)
    return name if name == str(label) else f"{name} [{label}]"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover(trace_dir: Path | None = None) -> dict[str, dict[str, Path]]:
    """Map split -> label -> file, for files named `<label>_<split>.jsonl`."""
    root = Path(trace_dir) if trace_dir is not None else TRACE_DIR
    found: dict[str, dict[str, Path]] = {}
    try:
        entries = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:
        return {}
    for path in entries:
        m = _FILE_RE.match(path.name)
        if not m or not path.is_file():
            continue
        label, split = m.group(1), m.group(2)
        found.setdefault(split, {})[label] = path
    return {s: found[s] for s in SPLITS if s in found}


def splits_available(trace_dir: Path | None = None) -> list[str]:
    return list(discover(trace_dir))


def labels_available(split: str, trace_dir: Path | None = None) -> list[str]:
    return sorted(discover(trace_dir).get(split, {}), key=label_sort_key)


# ---------------------------------------------------------------------------
# Trace cleaning. A trace file is data, so every field is checked.
# ---------------------------------------------------------------------------


def _int(v, default: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    return int(v)


def _float(v, default: float = 0.0) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    return float(v)


def _str(v, default: str = "") -> str:
    return v if isinstance(v, str) else default


def _opt_int(v):
    return None if isinstance(v, bool) or not isinstance(v, (int, float)) else int(v)


def clean_trace(rec) -> dict | None:
    """Return a normalised trace with safe types, or None if the record is unusable."""
    if not isinstance(rec, dict):
        return None
    task_id = rec.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return None
    steps: list[dict] = []
    raw_steps = rec.get("steps")
    for s in raw_steps if isinstance(raw_steps, list) else []:
        if not isinstance(s, dict):
            continue
        if s.get("type") == "tool_call":
            args = s.get("arguments")
            steps.append({
                "type": "tool_call",
                "name": _str(s.get("name"), "?"),
                "arguments": args if isinstance(args, dict) else {},
                "ok": s.get("ok") is True,
                "n_rows": _int(s.get("n_rows")),
                "vm_steps": _int(s.get("vm_steps")),
                "output": _str(s.get("output")),
            })
        elif s.get("type") == "final":
            steps.append({"type": "final", "text": _str(s.get("text"))})
    calls = [s for s in steps if s["type"] == "tool_call"]
    cost_in = rec.get("cost") if isinstance(rec.get("cost"), dict) else {}
    cost = {
        "tool_calls": _int(cost_in.get("tool_calls"), len(calls)),
        "sql_calls": _int(cost_in.get("sql_calls"), sum(1 for c in calls if c["name"] == "run_sql")),
        "failed_calls": _int(cost_in.get("failed_calls"), sum(1 for c in calls if not c["ok"])),
        "rows_returned": _int(cost_in.get("rows_returned"), sum(c["n_rows"] for c in calls)),
        "vm_steps": _int(cost_in.get("vm_steps"), sum(c["vm_steps"] for c in calls)),
    }
    opt_in = rec.get("optimal") if isinstance(rec.get("optimal"), dict) else {}
    optimal = {k: _opt_int(opt_in.get(k)) for k in ("tool_calls", "rows", "vm_steps")}
    predicted = rec.get("predicted")
    return {
        "task_id": task_id,
        "domain": _str(rec.get("domain")),
        "level": _int(rec.get("level")),
        "split": _str(rec.get("split")),
        "question": _str(rec.get("question")),
        "gold_answer": _str(rec.get("gold_answer")),
        "answer_type": _str(rec.get("answer_type")),
        "model": _str(rec.get("model")),
        "steps": steps,
        "predicted": predicted if isinstance(predicted, str) else None,
        "answered": rec.get("answered") is True,
        "grounded": rec.get("grounded") is True,
        "correct": rec.get("correct") is True,
        "efficiency": _float(rec.get("efficiency")),
        "reward": _float(rec.get("reward")),
        "latency_s": float(rec["latency_s"]) if _opt_int(rec.get("latency_s")) is not None else None,
        "cost": cost,
        "optimal": optimal,
    }


_CACHE: dict[tuple[str, int, int], dict[str, dict]] = {}


def load_traces(split: str, label: str, trace_dir: Path | None = None) -> dict[str, dict]:
    """task_id -> cleaned trace, for one (split, label). Empty if unknown or unreadable."""
    path = discover(trace_dir).get(split, {}).get(label)
    if path is None:
        return {}
    try:
        stat = path.stat()
        if stat.st_size > MAX_FILE_BYTES:
            return {}
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key in _CACHE:
            return _CACHE[key]
        out: dict[str, dict] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    trace = clean_trace(json.loads(line))
                except (ValueError, RecursionError):
                    continue
                if trace is not None:
                    out.setdefault(trace["task_id"], trace)
    except OSError:
        return {}
    _CACHE.clear()  # keep one generation only; the files are small
    _CACHE[key] = out
    return out


def load_split(split: str, trace_dir: Path | None = None) -> dict[str, dict[str, dict]]:
    """label -> task_id -> trace for one split."""
    return {label: load_traces(split, label, trace_dir) for label in labels_available(split, trace_dir)}


# ---------------------------------------------------------------------------
# Aggregates computed straight from the trace fields
# ---------------------------------------------------------------------------


def aggregate(traces: list[dict]) -> dict:
    n = len(traces)
    if n == 0:
        return {"tasks": 0}

    def mean(f) -> float:
        return sum(f(t) for t in traces) / n

    calls = sum(t["cost"]["tool_calls"] for t in traces)
    failed = sum(t["cost"]["failed_calls"] for t in traces)
    lat = [t["latency_s"] for t in traces if t["latency_s"] is not None]
    # The reward counts efficiency only for a correct answer, so this mean leaves the wrong ones out.
    eff_ok = [t["efficiency"] for t in traces if t["correct"]]
    return {
        "mean_efficiency_correct": (sum(eff_ok) / len(eff_ok)) if eff_ok else None,
        # same definition as frugal_analyst.metrics: correct, with no more calls than the oracle
        "correct_and_optimal_rate": mean(lambda t: 1.0 if t["correct"] and t["optimal"]["tool_calls"] is not None
                                         and t["cost"]["tool_calls"] <= t["optimal"]["tool_calls"] else 0.0),
        "tasks": n,
        # share of tool calls that did not fail, the same definition as the eval script's tool_call_validity
        "tool_call_validity": (1.0 - failed / calls) if calls else None,
        "mean_latency_s": (sum(lat) / len(lat)) if lat else None,
        "accuracy": mean(lambda t: 1.0 if t["correct"] else 0.0),
        "mean_tool_calls": mean(lambda t: t["cost"]["tool_calls"]),
        "mean_rows": mean(lambda t: t["cost"]["rows_returned"]),
        "mean_vm_steps": mean(lambda t: t["cost"]["vm_steps"]),
        "mean_efficiency": mean(lambda t: t["efficiency"]),
        "mean_reward": mean(lambda t: t["reward"]),
    }


def aggregates_by_label(split: str, trace_dir: Path | None = None) -> dict[str, dict]:
    """label -> aggregate for a split name, or for every split when split == 'all'."""
    out: dict[str, list[dict]] = {}
    for sp in (splits_available(trace_dir) if split == "all" else [split]):
        for label, traces in load_split(sp, trace_dir).items():
            out.setdefault(label, []).extend(traces.values())
    return {label: aggregate(ts) for label, ts in sorted(out.items(), key=lambda kv: label_sort_key(kv[0]))}


def default_pair(aggs: dict[str, dict]) -> tuple[str | None, str | None]:
    """Pick (before, after) labels for the first screen.

    zero-shot vs grpo, else zero-shot vs distilled. Else the worst and best main model by mean
    reward in the traces, else the worst and best label overall (scripted baselines).
    """
    labels = [l for l in aggs if aggs[l].get("tasks")]
    if not labels:
        return None, None
    if BASELINE_LABEL in labels:
        for after in PREFERRED_AFTER:
            if after in labels:
                return BASELINE_LABEL, after
    experiments = {"grpo-ungated", "zero-shot-stop", "distilled-mixed", "grpo-mixed", "distilled-all"}
    models = [l for l in labels if l not in SCRIPTED_LABELS]
    main = [l for l in models if l not in experiments]
    pool = main if len(main) >= 2 else models if len(models) >= 2 else labels
    ranked = sorted(pool, key=lambda l: (aggs[l]["mean_reward"], label_sort_key(l)))
    if len(ranked) == 1:
        return ranked[0], ranked[0]
    return ranked[0], ranked[-1]


# ---------------------------------------------------------------------------
# Task files (for the Ask tab)
# ---------------------------------------------------------------------------

_TASKS: dict | None = None


def example_tasks(task_dir: Path | None = None) -> dict[str, list]:
    """domain -> list of Task, from the fixed task files. Cached for the default dir."""
    global _TASKS
    use_cache = task_dir is None
    if use_cache and _TASKS is not None:
        return _TASKS
    from frugal_analyst.tasks import load_tasks

    root = Path(task_dir) if task_dir is not None else TASK_DIR
    by_domain: dict[str, list] = {}
    for split in EXAMPLE_SPLITS:
        path = root / f"{split}.jsonl"
        try:
            tasks = load_tasks(path) if path.is_file() else []
        except Exception:  # a bad task file must not stop the app
            tasks = []
        for t in tasks:
            by_domain.setdefault(t.domain, []).append(t)
    for ts in by_domain.values():
        ts.sort(key=lambda t: (t.level, t.id))
    if use_cache:
        _TASKS = by_domain
    return by_domain
