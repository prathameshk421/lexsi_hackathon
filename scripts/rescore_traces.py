"""Score old trace files again with the current reward. CPU only, standard library only.

    .venv/bin/python scripts/rescore_traces.py --traces outputs/traces/*.jsonl --out outputs/rescored/

For each trace, the script replays every recorded tool call against the local
databases (built by `scripts/gen_data.py --seed 0`). The databases are
deterministic, so a replayed call must give the same output, row count and VM
steps as the recorded call. If one call differs, the script stops and writes
nothing: the old trace then does not describe these databases.

One known exception: VM-step counts are not portable between machines
(probably because of the SQLite build). Traces from a model run on another
machine can differ from a local replay in VM steps only. `--allow-vm-drift` accepts that case (output text, ok flag and row
count must still match), uses the local count, and reports the drift.

Then it scores the episode with the current reward and writes a new file with
the same name to `--out`. It never changes the input files. It prints how many
traces changed `correct`, `grounded` or `reward`.

Exit codes: 0 ok, 1 a replayed call differs from the recorded call, 2 bad input.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python scripts/rescore_traces.py` finds frugal_analyst

from frugal_analyst.db import ensure_dbs  # noqa: E402
from frugal_analyst.reward import RewardConfig, ToolCall  # noqa: E402
from frugal_analyst.tasks import load_tasks  # noqa: E402
from frugal_analyst.traces import build_trace, read_traces, write_traces  # noqa: E402

EXIT_REPLAY_MISMATCH, EXIT_BAD_INPUT = 1, 2
WATCHED = ("correct", "grounded", "reward")
# These must be the same in the old trace and the task file, or the trace is for other data.
TASK_FIELDS = ("domain", "level", "split", "question", "gold_answer", "answer_type")


class InputError(Exception):
    pass


def load_task_index(tasks_dir: Path) -> dict:
    index = {}
    for path in sorted(tasks_dir.glob("*.jsonl")):
        for task in load_tasks(path):
            index[task.id] = task
    if not index:
        raise InputError(f"no tasks in {tasks_dir}/*.jsonl. Run scripts/gen_data.py --seed 0 first.")
    return index


def calls_and_final(trace: dict) -> tuple[list[ToolCall], str | None]:
    calls = [ToolCall(s["name"], s["arguments"]) for s in trace["steps"] if s.get("type") == "tool_call"]
    finals = [s["text"] for s in trace["steps"] if s.get("type") == "final"]
    return calls, (finals[-1] if finals else None)


def replay_problems(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Differences between the recorded and the replayed tool steps: (vm_steps only, everything else)."""
    vm_diffs, other = [], []
    old_steps, new_steps = old["steps"], new["steps"]
    if len(old_steps) != len(new_steps):
        return [], [f"{len(old_steps)} recorded steps but {len(new_steps)} replayed steps"]
    for i, (a, b) in enumerate(zip(old_steps, new_steps)):
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                text = f"steps[{i}].{key}: recorded {str(a.get(key))[:80]!r}, replayed {str(b.get(key))[:80]!r}"
                (vm_diffs if key == "vm_steps" else other).append(text)
    return vm_diffs, other


def rescore_record(old: dict, tasks: dict, cfg: RewardConfig, db_dir) -> tuple[dict, list[str], list[str]]:
    task = tasks.get(old.get("task_id"))
    if task is None:
        raise InputError(f"unknown task id {old.get('task_id')!r}")
    for key in TASK_FIELDS:
        if old.get(key) != getattr(task, key):
            raise InputError(f"{task.id}: trace field {key!r} is {old.get(key)!r} but the task file has {getattr(task, key)!r}")
    calls, final = calls_and_final(old)
    new = build_trace(task, calls, final, old["model"], raw_completion=old.get("raw_completion"),
                      completion_tokens=old.get("completion_tokens"), latency_s=old.get("latency_s"),
                      db_dir=db_dir, cfg=cfg)
    for key, value in old.items():  # keep any extra field a writer added
        new.setdefault(key, value)
    return (new, *replay_problems(old, new))


def changed(old: dict, new: dict, key: str) -> bool:
    a, b = old.get(key), new.get(key)
    if isinstance(a, float) or isinstance(b, float):
        return a is None or b is None or abs(float(a) - float(b)) > 1e-12
    return a != b


def summarize(name: str, olds: list[dict], news: list[dict], drift: list[bool]) -> dict:
    n = len(olds)
    row = {"file": name, "n": n}
    for key in WATCHED:
        row[f"{key}_changed"] = sum(changed(o, w, key) for o, w in zip(olds, news))
    # VM-step drift changes efficiency, so it changes the reward with no change to the rule.
    # Traces without drift show the effect of the rule alone.
    row["vm_drift_traces"] = sum(drift)
    row["reward_changed_without_vm_drift"] = sum(
        changed(o, w, "reward") for o, w, d in zip(olds, news, drift) if not d)
    row["correct_true_to_false"] = sum(bool(o["correct"]) and not w["correct"] for o, w in zip(olds, news))
    row["correct_false_to_true"] = sum(not o["correct"] and bool(w["correct"]) for o, w in zip(olds, news))
    row["accuracy_old"] = sum(bool(o["correct"]) for o in olds) / n
    row["accuracy_new"] = sum(bool(w["correct"]) for w in news) / n
    row["reward_old"] = sum(float(o["reward"]) for o in olds) / n
    row["reward_new"] = sum(float(w["reward"]) for w in news) / n
    row["answer_in_sql_literal"] = sum(bool(w["answer_in_sql_literal"]) for w in news)
    return row


def expand(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matches:
            raise InputError(f"no files match {pattern!r}")
        paths += [Path(m) for m in matches]
    paths = list(dict.fromkeys(paths))
    for p in paths:
        if not p.is_file():
            raise InputError(f"not a file: {p}")
    return paths


def run(args: argparse.Namespace) -> int:
    paths = expand(args.traces)
    out_dir = Path(args.out).resolve()
    if any(p.resolve().parent == out_dir for p in paths):
        raise InputError("--out is the folder of an input file. Pick another folder so the inputs stay as they are.")
    db_dir = args.db_dir
    ensure_dbs(db_dir)
    tasks = load_task_index(Path(args.tasks_dir))
    cfg = RewardConfig(mode=args.mode)

    results, problems, vm_problems = [], [], []
    for path in paths:
        olds = read_traces(path)
        news, drift = [], []
        for line_no, old in enumerate(olds, start=1):
            new, vm_diffs, other = rescore_record(old, tasks, cfg, db_dir)
            where = f"{path} record {line_no} ({old['task_id']})"
            problems += [f"{where}: {d}" for d in other]
            vm_problems += [f"{where}: {d}" for d in vm_diffs]
            news.append(new)
            drift.append(bool(vm_diffs))
        results.append((path, olds, news, drift))

    if vm_problems:
        print(f"{'WARNING' if args.allow_vm_drift else 'ERROR'}: {len(vm_problems)} replayed calls have other "
              f"VM steps than recorded (rows and output text are checked apart).", file=sys.stderr)
        for p in vm_problems[:20]:
            print("  " + p, file=sys.stderr)
    if problems or (vm_problems and not args.allow_vm_drift):
        if problems:
            print(f"ERROR: {len(problems)} replayed tool outputs differ from the recorded outputs.", file=sys.stderr)
            for p in problems[:20]:
                print("  " + p, file=sys.stderr)
        else:
            print("VM-step counts can change with the SQLite build. Use --allow-vm-drift to rescore with the "
                  "local counts.", file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        return EXIT_REPLAY_MISMATCH

    rows = []
    for path, olds, news, drift in results:
        write_traces(out_dir / path.name, news)
        rows.append(summarize(path.name, olds, news, drift))
    header = ["file", "n", "correct changed (T->F / F->T)", "grounded changed", "reward changed",
              "traces with VM drift", "reward changed, no VM drift",
              "accuracy old -> new", "mean reward old -> new", "answer_in_sql_literal"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["file"], str(r["n"]),
            f"{r['correct_changed']} ({r['correct_true_to_false']} / {r['correct_false_to_true']})",
            str(r["grounded_changed"]), str(r["reward_changed"]),
            str(r["vm_drift_traces"]), str(r["reward_changed_without_vm_drift"]),
            f"{r['accuracy_old']:.3f} -> {r['accuracy_new']:.3f}",
            f"{r['reward_old']:.4f} -> {r['reward_new']:.4f}",
            str(r["answer_in_sql_literal"]),
        ]) + " |")
    print(f"Replayed every tool call in {len(paths)} files. Output text, ok flags and row counts all match."
          + (f" {len(vm_problems)} calls differ in VM steps only." if vm_problems else " VM steps match too."))
    print(f"Reward mode: {args.mode}. Wrote {len(paths)} files to {out_dir}")
    print("\n".join(lines))
    (out_dir / "rescore_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="+", required=True, help="trace jsonl files (globs are expanded)")
    parser.add_argument("--out", required=True, help="output folder for the rescored files")
    parser.add_argument("--mode", default="gated", choices=["gated", "ungated"], help="reward mode")
    parser.add_argument("--tasks-dir", default=str(ROOT / "data" / "tasks"))
    parser.add_argument("--db-dir", default=None, help="database folder. Default: data/dbs or $FRUGAL_DB_DIR")
    parser.add_argument("--allow-vm-drift", action="store_true",
                        help="accept replayed calls whose only difference is the VM-step count, and use the local count")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (InputError, KeyError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())
