"""Handler logic for the four tabs. Plain functions, no Gradio import, so tests call them directly.

All SQL runs through `frugal_analyst.env`. This module opens no database itself.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import fa_data as data
import fa_render as render
from frugal_analyst.db import DOMAINS, ensure_dbs, schema_text
from frugal_analyst.env import MAX_ROWS_FETCHED, MAX_ROWS_SHOWN, execute_metered, format_value, run_tool
from frugal_analyst.policies import POLICIES
from frugal_analyst.reward import ToolCall
from frugal_analyst.traces import build_trace

MAX_QUESTION_CHARS = 300
MAX_SQL_CHARS = 2000
MAX_MODEL_CALLS = 8
MAX_TABLE_ROWS = 200
MAX_CELL_CHARS = 200
SCRIPTED_POLICIES = ("oracle", "naive")

LEVEL_CHOICES = [("All levels", "all"), ("Level 1", "1"), ("Level 2", "2"), ("Level 3", "3")]
NO_TRACES = (
    "No trace files found in outputs/traces. Run `python scripts/gen_data.py` to make the scripted ones. "
    "Model runs add more files named <label>_<split>.jsonl."
)


def init() -> None:
    """Build the SQLite files if they are missing. Called once at startup."""
    ensure_dbs()


def _log(exc: Exception) -> None:
    print(f"[app] {type(exc).__name__}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Compare (replay)
# ---------------------------------------------------------------------------


def split_choices(trace_dir: Path | None = None) -> list[str]:
    return data.splits_available(trace_dir)


def model_choices(split, trace_dir: Path | None = None) -> list[tuple[str, str]]:
    """(shown name, label). Scripted reference policies are marked so they are not mistaken for models."""
    return [(data.display_label(l), l) for l in data.labels_available(split, trace_dir)]


def _pair_for(split: str, trace_dir: Path | None = None) -> tuple[str | None, str | None]:
    return data.default_pair(data.aggregates_by_label(split, trace_dir))


def _valid_split(split, trace_dir: Path | None = None) -> str | None:
    avail = data.splits_available(trace_dir)
    if split in avail:
        return split
    return avail[0] if avail else None


def _level_value(level) -> int | None:
    return int(level) if str(level) in ("1", "2", "3") else None


def pair_candidates(split, label_a, label_b, domain="all", level="all", trace_dir=None) -> list[tuple[dict, dict]]:
    """Tasks that both models ran, after the domain and level filters. Sorted by task id."""
    split = _valid_split(split, trace_dir)
    if split is None:
        return []
    a = data.load_traces(split, str(label_a), trace_dir)
    b = data.load_traces(split, str(label_b), trace_dir)
    lv = _level_value(level)
    out = []
    for tid in sorted(set(a) & set(b)):
        ta, tb = a[tid], b[tid]
        if domain in DOMAINS and ta["domain"] != domain:
            continue
        if lv is not None and ta["level"] != lv:
            continue
        out.append((ta, tb))
    return out


def task_label(trace: dict) -> str:
    return f'{trace["task_id"]}  {render.clip(trace["question"], 80)}'


def task_choices(split, label_a, label_b, domain="all", level="all", trace_dir=None) -> list[tuple[str, str]]:
    return [(task_label(ta), ta["task_id"]) for ta, _ in pair_candidates(split, label_a, label_b, domain, level, trace_dir)]


def domain_choices(split, trace_dir=None) -> list[tuple[str, str]]:
    split = _valid_split(split, trace_dir)
    seen: set[str] = set()
    if split:
        for traces in data.load_split(split, trace_dir).values():
            seen.update(t["domain"] for t in traces.values())
    return [("All domains", "all")] + [(d, d) for d in DOMAINS if d in seen]


def difference_score(ta: dict, tb: dict) -> tuple:
    """Bigger means the two runs differ more: reward first, then rows, VM steps and calls."""
    return (
        abs(ta["reward"] - tb["reward"]),
        abs(ta["cost"]["rows_returned"] - tb["cost"]["rows_returned"]),
        abs(ta["cost"]["vm_steps"] - tb["cost"]["vm_steps"]),
        abs(ta["cost"]["tool_calls"] - tb["cost"]["tool_calls"]),
    )


def most_different_task(split, label_a, label_b, domain="all", level="all", trace_dir=None) -> str | None:
    cands = pair_candidates(split, label_a, label_b, domain, level, trace_dir)
    if not cands:
        return None
    best = max(cands, key=lambda p: (difference_score(*p), p[0]["task_id"]))
    return best[0]["task_id"]


def random_task(split, label_a, label_b, domain="all", level="all", trace_dir=None, rng=None) -> str | None:
    cands = pair_candidates(split, label_a, label_b, domain, level, trace_dir)
    if not cands:
        return None
    return (rng or random).choice(cands)[0]["task_id"]


def replay_view(split, label_a, label_b, task_id, trace_dir=None) -> str:
    """HTML for the two trajectories side by side. Never raises."""
    try:
        split = _valid_split(split, trace_dir)
        if split is None:
            return render.notice_html(NO_TRACES, "info", "No traces yet")
        labels = data.labels_available(split, trace_dir)
        if label_a not in labels or label_b not in labels:
            return render.notice_html("Pick two models from the lists above.", "info")
        ta = data.load_traces(split, label_a, trace_dir).get(task_id)
        tb = data.load_traces(split, label_b, trace_dir).get(task_id)
        if ta is None or tb is None:
            return render.notice_html(
                "Both models must have run this task. Pick a task from the list, or change the filters.", "info",
                "No task selected",
            )
        return render.replay_html(ta, str(label_a), tb, str(label_b))
    except Exception as exc:
        _log(exc)
        return render.notice_html("This comparison could not be shown.", "bad", "Something went wrong")


def initial_compare_state(trace_dir=None) -> dict:
    """Values for the first screen, so it already shows a real comparison."""
    splits = split_choices(trace_dir)
    if not splits:
        return {"splits": [], "split": None, "labels": [], "a": None, "b": None, "domains": [("All domains", "all")],
                "tasks": [], "task": None}
    split = "test_id" if "test_id" in splits else splits[0]
    a, b = _pair_for(split, trace_dir)
    tasks = task_choices(split, a, b, "all", "all", trace_dir)
    return {
        "splits": splits, "split": split, "labels": model_choices(split, trace_dir), "a": a, "b": b,
        "domains": domain_choices(split, trace_dir), "tasks": tasks,
        "task": most_different_task(split, a, b, trace_dir=trace_dir) or (tasks[0][1] if tasks else None),
    }


def on_split_change(split, trace_dir=None) -> dict:
    """New model labels, domains and task list for a split."""
    split = _valid_split(split, trace_dir)
    if split is None:
        return initial_compare_state(trace_dir)
    a, b = _pair_for(split, trace_dir)
    tasks = task_choices(split, a, b, "all", "all", trace_dir)
    return {
        "split": split, "labels": model_choices(split, trace_dir), "a": a, "b": b,
        "domains": domain_choices(split, trace_dir), "tasks": tasks,
        "task": most_different_task(split, a, b, trace_dir=trace_dir) or (tasks[0][1] if tasks else None),
    }


def on_filter_change(split, label_a, label_b, domain, level, current_task, trace_dir=None) -> tuple[list, str | None]:
    """Task choices after a filter or model change. Keeps the current task when it is still listed."""
    choices = task_choices(split, label_a, label_b, domain, level, trace_dir)
    ids = [v for _, v in choices]
    if current_task in ids:
        return choices, current_task
    pick = most_different_task(split, label_a, label_b, domain, level, trace_dir)
    return choices, pick


# ---------------------------------------------------------------------------
# Ask (live)
# ---------------------------------------------------------------------------


def valid_database(database) -> str | None:
    return database if isinstance(database, str) and database in DOMAINS else None


def schema_for(database) -> str:
    db = valid_database(database)
    if db is None:
        return "Pick a database."
    try:
        return schema_text(db)
    except Exception as exc:
        _log(exc)
        return "The schema could not be read."


def example_choices(database) -> list[tuple[str, str]]:
    db = valid_database(database)
    if db is None:
        return []
    return [(f"L{t.level}  {render.clip(t.question, 90)}", t.id) for t in data.example_tasks().get(db, [])]


def example_task(database, task_id):
    db = valid_database(database)
    for t in data.example_tasks().get(db, []) if db else []:
        if t.id == task_id:
            return t
    return None


def question_for_example(database, task_id) -> str:
    t = example_task(database, task_id)
    return t.question if t else ""


def task_for_question(database, question):
    """The task whose question text is exactly this one, or None for a typed question."""
    db = valid_database(database)
    q = (question or "").strip()
    for t in data.example_tasks().get(db, []) if db else []:
        if t.question == q:
            return t
    return None


def policy_choices(backend) -> list[tuple[str, str]]:
    choices = [
        ("Oracle: the efficient scripted path", "oracle"),
        ("Naive: scripted, right but wasteful", "naive"),
    ]
    if backend is not None and backend.available():
        choices.append((f"Model: {backend.name}", "model"))
    return choices


def backend_note(backend) -> str:
    if backend is not None and backend.available():
        return "Live model inference is on. Scripted policies only work for the example questions."
    return (
        "Live model inference is off on this deployment, and replay is on. Oracle and naive run only "
        "on the example questions, because they need the gold path."
    )


def normalize_calls(calls) -> list[ToolCall]:
    out: list[ToolCall] = []
    for c in list(calls or [])[:MAX_MODEL_CALLS]:
        if isinstance(c, ToolCall):
            out.append(ToolCall(str(c.name), c.arguments if isinstance(c.arguments, dict) else {}))
        elif isinstance(c, dict):
            args = c.get("arguments")
            out.append(ToolCall(str(c.get("name", "?")), args if isinstance(args, dict) else {}))
    return out


def free_trace(database: str, question: str, calls: list[ToolCall], final_text: str | None, label: str) -> dict:
    """Trace for a typed question. No gold answer, so there is no verdict and no efficiency."""
    steps, rows, vm, failed, sql_calls = [], 0, 0, 0, 0
    for c in calls:
        r = run_tool(c.name, c.arguments)
        steps.append({"type": "tool_call", "name": c.name, "arguments": c.arguments, "ok": r.ok,
                      "n_rows": r.n_rows, "vm_steps": r.vm_steps, "output": r.output})
        rows += r.n_rows
        vm += r.vm_steps
        failed += 0 if r.ok else 1
        sql_calls += 1 if r.is_sql else 0
    if final_text is not None:
        steps.append({"type": "final", "text": final_text})
    return data.clean_trace({
        "task_id": "typed", "domain": database, "level": 0, "split": "live", "question": question,
        "gold_answer": "", "answer_type": "", "model": label, "steps": steps,
        "predicted": None, "answered": final_text is not None, "grounded": False, "correct": False,
        "efficiency": 0.0, "reward": 0.0,
        "cost": {"tool_calls": len(calls), "sql_calls": sql_calls, "failed_calls": failed,
                 "rows_returned": rows, "vm_steps": vm},
        "optimal": {},
    })


def ask_run(database, question, policy, backend=None, pace: bool = True):
    """Generator of HTML pages. Each step of the run appears one after the other."""
    try:
        yield from _ask_run(database, question, policy, backend, pace)
    except Exception as exc:
        _log(exc)
        yield render.notice_html("This run could not finish. Try another question or policy.", "bad", "Something went wrong")


def _ask_run(database, question, policy, backend, pace):
    db = valid_database(database)
    if db is None:
        yield render.notice_html("Pick a database from the list.", "info")
        return
    q = (question or "").strip()
    if not q:
        yield render.notice_html("Pick an example question or type your own, then press Run.", "info")
        return
    if len(q) > MAX_QUESTION_CHARS:
        yield render.notice_html(f"The question is too long. The limit is {MAX_QUESTION_CHARS} characters.", "bad")
        return
    task = task_for_question(db, q)
    model_ok = backend is not None and backend.available()
    if policy in SCRIPTED_POLICIES:
        if task is None:
            msg = (
                f"The {policy} policy needs the gold path, so it only works for the example questions. "
                "Pick an example, or use the model policy for a question you typed."
                if model_ok else
                f"The {policy} policy needs the gold path, so it only works for the example questions. "
                "Pick an example from the list. Typed questions need a live model, which is off here."
            )
            yield render.notice_html(msg, "warn", "Typed question")
            return
        calls, final_text = POLICIES[policy](task)
        trace = build_trace(task, calls, final_text, policy)
        label, has_gold, note = policy, True, ""
    elif policy == "model":
        if not model_ok:
            yield render.notice_html("Live model inference is off on this deployment. Replay is on.", "info")
            return
        yield render.notice_html("The model is working. The first run can take a while.", "info", "Running")
        calls, final_text, _raw = backend.run(db, q)
        calls = normalize_calls(calls)
        label = backend.name
        if task is not None:
            trace = build_trace(task, calls, final_text, label)
            has_gold, note = True, ""
        else:
            trace = free_trace(db, q, calls, final_text, label)
            has_gold, note = False, "Typed question: cost is shown, there is no gold answer to check."
    else:
        yield render.notice_html("Pick a policy.", "info")
        return

    n = len([s for s in trace["steps"] if s["type"] == "tool_call"])
    trace = data.clean_trace(trace)  # same cleaning as replay, so both views render the same way
    delay = min(0.45, 4.0 / max(n, 1)) if pace else 0.0
    yield render.live_html(trace, label, 0, n == 0, has_gold, q, note)
    for k in range(1, n + 1):
        if delay:
            time.sleep(delay)
        yield render.live_html(trace, label, k, k == n, has_gold, q, note)


# ---------------------------------------------------------------------------
# SQL playground
# ---------------------------------------------------------------------------

# Each pair gives the same answer two ways. The numbers are measured when the pair runs.
EXAMPLE_PAIRS = [
    {
        "name": "Count rows: COUNT or dump the table",
        "database": "retail",
        "cheap": "SELECT COUNT(*) FROM products",
        "costly": "SELECT * FROM products",
        "same": "row_count",  # the costly query gives the answer as its number of rows
    },
    {
        "name": "Maximum fee: MAX or sort every fee",
        "database": "clinic",
        "cheap": "SELECT MAX(fee) FROM appointments",
        "costly": "SELECT fee FROM appointments ORDER BY fee DESC",
    },
    {
        "name": "Count shipments: direct or with needless joins",
        "database": "logistics",
        "cheap": "SELECT COUNT(*) FROM shipments",
        "costly": (
            "SELECT COUNT(*) FROM shipments s JOIN depots d ON d.depot_id = s.depot_id "
            "JOIN carriers c ON c.carrier_id = s.carrier_id"
        ),
    },
]


def pair_choices() -> list[tuple[str, str]]:
    return [(p["name"], str(i)) for i, p in enumerate(EXAMPLE_PAIRS)]


def _pair(index):
    try:
        return EXAMPLE_PAIRS[int(index)]
    except (ValueError, TypeError, IndexError):
        return None


def load_pair(index) -> tuple[str, str]:
    """(database, cheap query) to put in the editor."""
    p = _pair(index)
    return (p["database"], p["cheap"]) if p else (DOMAINS[0], "")


def cost_hint(n_rows: int, capped: bool) -> str:
    if n_rows > MAX_ROWS_SHOWN:
        shown = f"{n_rows}+" if capped else str(n_rows)
        return (
            f"This would cost {shown} rows of context. An agent sees {MAX_ROWS_SHOWN} of them and still pays "
            "for the rest. An aggregate such as COUNT or SUM, or a LIMIT, keeps it small."
        )
    return ""


def _table(columns: list[str], rows: list[tuple]):
    import pandas as pd

    shown = rows[:MAX_TABLE_ROWS]
    body = [[render.clip(format_value(v), MAX_CELL_CHARS) for v in r] for r in shown]
    cols = [render.clip(c, 80) for c in columns] or ["result"]
    return pd.DataFrame(body, columns=cols)


def _empty_table(message: str = "no result"):
    import pandas as pd

    return pd.DataFrame([], columns=[message])


def playground_run(database, sql):
    """Returns (stats html, result table). All SQL goes through env.execute_metered."""
    db = valid_database(database)
    if db is None:
        return render.notice_html("Pick a database from the list.", "info"), _empty_table()
    text = sql if isinstance(sql, str) else ""
    if not text.strip():
        return render.notice_html("Type a SELECT query, or load an example pair below.", "info"), _empty_table()
    if len(text) > MAX_SQL_CHARS:
        return render.notice_html(f"The query is too long. The limit is {MAX_SQL_CHARS} characters.", "bad"), _empty_table()
    try:
        res = execute_metered(db, text)
    except Exception as exc:  # execute_metered should never raise. This is a last guard.
        _log(exc)
        return render.notice_html("The query could not run.", "bad"), _empty_table()
    if not res.ok:
        return render.query_stats_html(0, res.vm_steps, False, "", res.error or "unknown error"), _empty_table("no rows")
    capped = res.n_rows >= MAX_ROWS_FETCHED
    extra = f"The table shows the first {MAX_TABLE_ROWS} rows." if res.n_rows > MAX_TABLE_ROWS else ""
    stats = render.query_stats_html(res.n_rows, res.vm_steps, capped, cost_hint(res.n_rows, capped), note=extra)
    return stats, _table(res.columns, res.rows)


def run_pair(index) -> str:
    p = _pair(index)
    if p is None:
        return render.notice_html("Pick an example pair.", "info")
    items = []
    for title, key in (("Cheap", "cheap"), ("Costly", "costly")):
        res = execute_metered(p["database"], p[key])
        items.append((title, p[key], res.n_rows, res.vm_steps, None if res.ok else (res.error or "failed")))
    return render.pair_html(items)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

HEADLINE_KEYS = [
    "accuracy", "mean_reward", "mean_efficiency", "mean_tool_calls", "mean_rows", "mean_rows_returned",
    "mean_vm_steps", "grounded_rate", "answered_rate", "correct_and_optimal_rate", "vm_step_ratio",
]


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def read_results(path: Path | None = None):
    """Parsed results.json, or None when it is absent or unreadable."""
    p = Path(path) if path is not None else data.RESULTS_PATH
    try:
        if not p.is_file() or p.stat().st_size > 20 * 1024 * 1024:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None


def normalize_results(obj, max_depth: int = 5, max_rows: int = 200) -> list[dict]:
    """Flatten any nested results object into rows. A row is a dict of numbers found together.

    The first column `group` is the path of keys, for example "test_id / grpo".
    """
    rows: list[dict] = []

    def walk(node, path: list[str], depth: int) -> None:
        if depth > max_depth or len(rows) >= max_rows:
            return
        if isinstance(node, dict):
            nums = {str(k): (round(v, 4) if isinstance(v, float) else v) for k, v in node.items() if _is_num(v)}
            if len(nums) >= 2:
                rows.append({"group": " / ".join(path) or "all", **nums})
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, path + [str(k)], depth + 1)
        elif isinstance(node, list):
            for i, v in enumerate(node[:max_rows]):
                if isinstance(v, dict):
                    ident = [v[k] for k in ("split", "model", "label", "name") if isinstance(v.get(k), str)]
                    walk(v, path + (ident or [str(i)]), depth + 1)

    walk(obj, [], 0)
    return rows


def _row_model(group: str) -> tuple[str | None, frozenset]:
    tokens = [t.strip() for t in group.split("/")]
    models = [t for t in tokens if t in data.LABEL_ORDER]
    if len(models) != 1:
        return None, frozenset(tokens)
    return models[0], frozenset(t for t in tokens if t != models[0])


def headline_from_rows(rows: list[dict]):
    """(before, after, cards) from results rows when a zero-shot row and a later row share a context."""
    by_model: dict[str, dict[frozenset, dict]] = {}
    for r in rows:
        model, ctx = _row_model(r["group"])
        if model:
            by_model.setdefault(model, {})[ctx] = r
    for before, after in (("zero-shot", "grpo"), ("zero-shot", "distilled")):
        if before in by_model and after in by_model:
            shared = [c for c in by_model[before] if c in by_model[after]]
            if not shared:
                continue
            shared.sort(key=lambda c: (("test_id" not in c), sorted(c)))
            rb, ra = by_model[before][shared[0]], by_model[after][shared[0]]
            keys = [k for k in HEADLINE_KEYS if _is_num(rb.get(k)) and _is_num(ra.get(k))]
            return before, after, [(k, rb[k], ra[k]) for k in keys[:6]]
    return None


EVAL_HEADLINE = ["accuracy", "reward", "efficiency", "tool_calls", "rows_returned", "vm_steps"]
KEY_KEYS = ["accuracy", "tool_call_validity", "reward", "efficiency", "tool_calls", "rows_returned", "vm_steps", "latency_s"]
SHORT_HEADERS = {
    "model": "Model", "split": "Split", "n": "Tasks", "accuracy": "Accuracy", "tool_call_validity": "Tool-call validity",
    "reward": "Reward", "efficiency": "Efficiency", "tool_calls": "Tool calls", "rows_returned": "Rows",
    "vm_steps": "VM steps", "latency_s": "Latency (s)",
}
PREFIX_KEYS = ("model", "split", "n")
NON_METRIC = {"model", "split", "n", "name", "raw", "kind"}
DEFAULT_CHART_KEYS = ["efficiency", "tool_calls"]
SCRIPTED_NOTE = "naive and oracle are scripted reference policies, not models."


def display_label(label: str) -> str:
    return f"{label} (scripted)" if label in data.SCRIPTED_LABELS else label


def _trace_rows(split: str, trace_dir=None) -> list[dict]:
    aggs = data.aggregates_by_label(split, trace_dir)
    rows = []
    for label, a in aggs.items():
        if not a.get("tasks"):
            continue
        row = {"model": display_label(label), "raw": label, "n": a["tasks"], "name": display_label(label),
               "kind": "scripted reference" if label in data.SCRIPTED_LABELS else "model"}
        for key, src in (("accuracy", "accuracy"), ("tool_call_validity", "tool_call_validity"), ("reward", "mean_reward"),
                         ("efficiency", "mean_efficiency"), ("tool_calls", "mean_tool_calls"), ("rows_returned", "mean_rows"),
                         ("vm_steps", "mean_vm_steps"), ("latency_s", "mean_latency_s")):
            if _is_num(a.get(src)):
                row[key] = round(a[src], 3)
        rows.append(row)
    return rows


def _build_view(rows: list[dict], source: str, note: str, headline: str, metric_label=None) -> dict:
    """Common shape for both sources: a short table, a full table and the chart metrics."""
    present = {k for r in rows for k in r if k not in NON_METRIC and _is_num(r.get(k))}
    keys = [k for k in KEY_KEYS if k in present]
    others = sorted(present - set(KEY_KEYS))

    def header(k: str) -> str:
        if k in SHORT_HEADERS:
            return SHORT_HEADERS[k]
        return metric_label(k) if metric_label else render.pretty_key(k)

    prefix = [k for k in PREFIX_KEYS if any(r.get(k) not in (None, "") for r in rows)]
    short = [(k, header(k)) for k in prefix + keys]
    full = [(k, header(k)) for k in prefix + keys + others]
    if any(r.get("raw") in data.SCRIPTED_LABELS for r in rows):
        note += " " + SCRIPTED_NOTE
    return {
        "source": source, "note": note, "headers": short, "all_headers": full, "rows": rows, "headline": headline,
        "metrics": [header(k) for k in keys + others], "metric_keys": {header(k): k for k in keys + others}, "x": "name",
    }


def default_metric(view: dict) -> str | None:
    """A telling default: mean efficiency, else mean tool calls, else the first metric."""
    for key in DEFAULT_CHART_KEYS:
        for header, k in view["metric_keys"].items():
            if k == key:
                return header
    return view["metrics"][0] if view["metrics"] else None


def parse_eval(obj, split: str = "all") -> dict | None:
    """Read the shape written by scripts/eval_auditkit.py: splits -> split -> models -> label -> metrics.

    Returns a results view, or None if the object does not have that shape.
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("splits"), dict):
        return None
    info = obj.get("metric_info") if isinstance(obj.get("metric_info"), dict) else {}

    def metric_label(key: str) -> str:
        entry = info.get(key)
        label = entry.get("label") if isinstance(entry, dict) else None
        return render.pretty_key(render.clip(label, 40) if isinstance(label, str) and label else key)

    all_splits = [s for s, v in obj["splits"].items() if isinstance(s, str) and isinstance(v, dict)]
    chosen = [split] if split in all_splits else all_splits
    rows: list[dict] = []
    for sp in chosen:
        models = obj["splits"][sp].get("models")
        if not isinstance(models, dict):
            continue
        for label in (m for m in models if isinstance(m, str)):
            entry = models[label]
            metrics = entry.get("metrics") if isinstance(entry, dict) else None
            if not isinstance(metrics, dict):
                continue
            row = {"model": display_label(label), "raw": label, "split": sp, "n": entry.get("n") if _is_num(entry.get("n")) else ""}
            row.update({str(k): (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items() if _is_num(v)})
            rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda r: (data.label_sort_key(r["raw"]), r["split"]))
    for r in rows:
        r["name"] = r["model"] if len(chosen) == 1 else f'{r["model"]} / {r["split"]}'
        r["kind"] = "scripted reference" if r["raw"] in data.SCRIPTED_LABELS else "model"
    note = "Numbers from outputs/eval/results.json." + (f" Split: {chosen[0]}." if len(chosen) == 1 else "")
    return _build_view(rows, "results", note, _eval_headline(obj, chosen, metric_label), metric_label)


def _eval_headline(obj: dict, chosen: list[str], metric_label) -> str:
    """Before and after cards from the `comparisons` block, with its bootstrap interval."""
    comps = obj.get("comparisons")
    if not isinstance(comps, dict):
        return ""
    sp = "test_id" if "test_id" in chosen and "test_id" in comps else next((s for s in chosen if s in comps), None)
    block = comps.get(sp) if sp else None
    pairs = block.get("pairs") if isinstance(block, dict) else None
    if not isinstance(pairs, list) or not pairs:
        return ""
    pair = next((p for p in pairs if isinstance(p, dict) and p.get("model") == "grpo"), None) or next(
        (p for p in pairs if isinstance(p, dict) and isinstance(p.get("metrics"), dict)), None)
    if pair is None or not isinstance(pair.get("metrics"), dict):
        return ""
    cards = []
    for key in EVAL_HEADLINE:
        m = pair["metrics"].get(key)
        if not isinstance(m, dict) or not _is_num(m.get("baseline_mean")) or not _is_num(m.get("candidate_mean")):
            continue
        extra = ""
        if _is_num(m.get("ci_low")) and _is_num(m.get("ci_high")):
            extra = f'95% interval of the change: {m["ci_low"]:.3g} to {m["ci_high"]:.3g}'
        if isinstance(m.get("verdict"), str):
            extra = (extra + f' ({render.clip(m["verdict"], 20)})').strip()
        cards.append((metric_label(key), m["baseline_mean"], m["candidate_mean"], extra))
    base = block.get("baseline") if isinstance(block.get("baseline"), str) else "baseline"
    return render.headline_html(display_label(base), display_label(str(pair.get("model", "candidate"))), cards,
                                f"split {sp}, from results.json")


def results_split_choices(trace_dir=None) -> list[str]:
    return ["all"] + data.splits_available(trace_dir)


def _empty_view(note: str) -> dict:
    return {"source": "none", "note": note, "headers": [], "all_headers": [], "rows": [], "headline": "",
            "metrics": [], "metric_keys": {}, "x": "name"}


def results_view(split="all", trace_dir=None, results_path=None) -> dict:
    """Everything the Results tab needs. `source` says where the numbers come from."""
    try:
        raw = read_results(results_path)
        parsed = parse_eval(raw, split) if raw is not None else None
        if parsed is not None:
            return parsed
        rows = normalize_results(raw) if raw is not None else []
        if rows:
            for r in rows:
                r["name"] = r["group"]
                r["raw"] = ""
            view = _build_view(rows, "results", "Numbers from outputs/eval/results.json.", "")
            rest = [(k, h) for k, h in view["all_headers"] if k not in PREFIX_KEYS]
            view["headers"] = [("group", "Group")] + rest[:8]
            view["all_headers"] = [("group", "Group")] + rest
            head = headline_from_rows(rows)
            view["headline"] = render.headline_html(head[0], head[1], head[2], "from results.json") if head else ""
            return view
        note = "No evaluation results yet. Showing numbers computed from the trace files."
        if raw is not None:
            note = "outputs/eval/results.json has a shape this page cannot read. Showing numbers computed from the trace files."
        split = split if split in results_split_choices(trace_dir) else "all"
        rows = _trace_rows(split, trace_dir)
        if not rows:
            return _empty_view("No results yet. Run the evaluation or make trace files, then reload.")
        aggs = data.aggregates_by_label(split, trace_dir)
        a, b = data.default_pair(aggs)
        cards = []
        if a and b and a != b:
            cards = [(k, aggs[a][k], aggs[b][k]) for k in
                     ("accuracy", "mean_tool_calls", "mean_rows", "mean_vm_steps", "mean_efficiency", "mean_reward")]
        headline = render.headline_html(display_label(a), display_label(b), cards, "computed from traces") if cards else ""
        return _build_view(rows, "traces", f"{note} Split: {split}.", headline)
    except Exception as exc:
        _log(exc)
        return _empty_view("The results could not be read.")


def results_table(view: dict, full: bool = False):
    import pandas as pd

    headers = view["all_headers"] if full else view["headers"]
    if not view["rows"] or not headers:
        return pd.DataFrame([], columns=["no results"])

    def cell(r, k):
        v = r.get(k, "")
        return round(v, 3) if isinstance(v, float) else v

    return pd.DataFrame([[cell(r, k) for k, _ in headers] for r in view["rows"]], columns=[h for _, h in headers])


def table_widths(df) -> list[int]:
    """Minimum column widths in pixels, so headers never wrap one letter per line."""
    return [max(200 if i == 0 else 96, min(len(str(c)) * 9 + 36, 240)) for i, c in enumerate(df.columns)]


def results_chart(view: dict, metric):
    """Frame (name, value, group) for one metric, or an empty frame."""
    import pandas as pd

    key = view["metric_keys"].get(metric)
    empty = pd.DataFrame({"name": [], "value": [], "group": []})
    if not view["rows"] or key is None:
        return empty
    rows = [r for r in view["rows"] if _is_num(r.get(key))]
    multi = len({r.get("split") for r in rows if r.get("split")}) > 1
    return pd.DataFrame({
        "name": [str(r.get("name", "")) for r in rows],
        "value": [r[key] for r in rows],
        "group": [(r.get("split") if multi else r.get("kind", "model")) or "all" for r in rows],
    })


def axis_upper(df) -> float:
    """Top of the value axis. The axis always starts at 0. Rates stop at 1, counts get 10 percent headroom."""
    top = float(df["value"].max()) if len(df) else 1.0
    return 1.0 if 0 < top <= 1.0 else max(top * 1.1, 1.0)


def chart_html(view: dict, metric) -> str:
    """The results chart as plain HTML bars. It has no fixed pixel width, so it fits a phone."""
    df = results_chart(view, metric)
    if not len(df):
        return render.notice_html("No chart yet. It appears when there are results.", "info")
    scripted = {str(r.get("name", "")): r.get("raw") in data.SCRIPTED_LABELS for r in view["rows"]}
    items = [(n, v, g, scripted.get(n, False)) for n, v, g in zip(df["name"], df["value"], df["group"])]
    return render.chart_html(metric or "", items, axis_upper(df))
