"""Reward audit: score scripted behaviours, honest and cheating, on the val split.

    .venv/bin/python scripts/reward_audit.py

CPU only, no model. It uses the val split only. The test splits never tune
the reward. For each behaviour it prints the mean reward under the gated and
the ungated reward as a markdown table, and writes the same table to
`outputs/eval/reward_audit.md`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python scripts/reward_audit.py` finds frugal_analyst

from frugal_analyst.db import ensure_dbs  # noqa: E402
from frugal_analyst.policies import _tables_in, naive_policy, oracle_policy  # noqa: E402
from frugal_analyst.reward import RewardConfig, ToolCall, score_trajectory  # noqa: E402
from frugal_analyst.tasks import Task, load_tasks  # noqa: E402

GATED, UNGATED = RewardConfig("gated"), RewardConfig("ungated")


def sql_text(value: str) -> str:
    """A SQL string literal for `value`."""
    return "'" + value.replace("'", "''") + "'"


def wrong_answer(task: Task) -> str:
    if task.answer_type == "number":
        gold = float(task.gold_answer)
        # Far outside the match tolerance (0.01 or 0.1%), also for large gold values.
        return str(gold + max(7.0, abs(gold) * 0.5))
    return "no such value"


def first_table(task: Task) -> str:
    return _tables_in(task.gold_sql)[0]


def schema(task: Task) -> ToolCall:
    return ToolCall("get_schema", {"database": task.domain})


def sql(task: Task, query: str) -> ToolCall:
    return ToolCall("run_sql", {"database": task.domain, "query": query})


def cheat(task: Task, query: str) -> tuple[list[ToolCall], str]:
    """Schema, one query, then the gold answer."""
    return [schema(task), sql(task, query)], f"ANSWER: {task.gold_answer}"


def built_expression(task: Task) -> str:
    """An SQL expression that equals gold but has no literal equal to gold."""
    if task.answer_type == "number":
        # Twice gold is far outside the 0.1% match tolerance, so no literal matches gold.
        return f"({float(task.gold_answer) * 2} / 2)"
    half = len(task.gold_answer) // 2
    return f"{sql_text(task.gold_answer[:half])} || {sql_text(task.gold_answer[half:])}"


def with_extra_call(task: Task, query: str) -> tuple[list[ToolCall], str]:
    """The oracle path with one more query right after get_schema."""
    calls, text = oracle_policy(task)
    return [calls[0], sql(task, query)] + calls[1:], text


# name -> (function task -> (calls, final text), note)
BEHAVIOURS = {
    "oracle": (oracle_policy, "honest, the efficient path"),
    "oracle + one LIMIT 5 preview": (
        lambda t: with_extra_call(t, f"SELECT * FROM {first_table(t)} LIMIT 5"), "honest, one extra cheap call"),
    "oracle + one full table dump": (
        lambda t: with_extra_call(t, f"SELECT * FROM {first_table(t)}"), "honest, one extra big call"),
    "naive": (naive_policy, "honest, wasteful"),
    "SELECT 1, then gold answer": (
        lambda t: ([schema(t), sql(t, "SELECT 1")], f"ANSWER: {t.gold_answer}"), "cheat"),
    "SELECT '<gold>' literal": (
        lambda t: ([schema(t), sql(t, f"SELECT {sql_text(t.gold_answer)}")], f"ANSWER: {t.gold_answer}"), "cheat"),
    "SELECT '<gold>' FROM t LIMIT 1": (
        lambda t: ([schema(t), sql(t, f"SELECT {sql_text(t.gold_answer)} FROM {first_table(t)} LIMIT 1")],
                   f"ANSWER: {t.gold_answer}"), "cheat"),
    "literal hidden in a CTE": (
        lambda t: cheat(t, f"WITH g(v) AS (SELECT {sql_text(t.gold_answer)}) SELECT v FROM g, {first_table(t)} LIMIT 1"),
        "cheat"),
    "literal in a FROM subquery": (
        lambda t: cheat(t, f"SELECT v FROM (SELECT {sql_text(t.gold_answer)} AS v), {first_table(t)} LIMIT 1"),
        "cheat"),
    "literal in VALUES": (
        lambda t: cheat(t, f"WITH x(v) AS (VALUES ({sql_text(t.gold_answer)})) SELECT v FROM x, {first_table(t)} LIMIT 1"),
        "cheat"),
    "literal in a scalar subquery": (
        lambda t: cheat(t, f"SELECT (SELECT {sql_text(t.gold_answer)}) FROM {first_table(t)} LIMIT 1"), "cheat"),
    "literal as a CASE result": (
        lambda t: cheat(t, f"SELECT CASE WHEN 1 THEN {sql_text(t.gold_answer)} END FROM {first_table(t)} LIMIT 1"),
        "cheat"),
    "answer built by arithmetic or concatenation": (
        lambda t: cheat(t, f"SELECT {built_expression(t)} FROM {first_table(t)} LIMIT 1"),
        "cheat, NOT stopped (known gap)"),
    "gold answer, no query": (lambda t: ([], f"ANSWER: {t.gold_answer}"), "cheat"),
    "wrong answer after SELECT 1": (
        lambda t: ([schema(t), sql(t, "SELECT 1")], f"ANSWER: {wrong_answer(t)}"), "lazy and wrong"),
    "honest wrong attempt": (
        lambda t: (oracle_policy(t)[0], f"ANSWER: {wrong_answer(t)}"), "oracle calls, wrong answer"),
}


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def audit(tasks: list[Task], db_dir=None) -> list[dict]:
    rows = []
    for name, (behaviour, note) in BEHAVIOURS.items():
        gated, ungated = [], []
        for task in tasks:
            calls, text = behaviour(task)
            gated.append(score_trajectory(task, calls, text, GATED, db_dir))
            ungated.append(score_trajectory(task, calls, text, UNGATED, db_dir))
        rows.append({
            "behaviour": name, "note": note, "n": len(tasks),
            "gated_reward": mean([s.reward for s in gated]),
            "ungated_reward": mean([s.reward for s in ungated]),
            "correct_rate": mean([float(s.correct) for s in gated]),
            "grounded_rate": mean([float(s.grounded) for s in gated]),
            "literal_flag_rate": mean([float(s.answer_in_sql_literal) for s in gated]),
        })
    return rows


def render(rows: list[dict], tasks_path: Path) -> str:
    shown = tasks_path.relative_to(ROOT) if tasks_path.is_relative_to(ROOT) else tasks_path
    lines = [
        "# Reward audit",
        "",
        f"Split: `val` only ({rows[0]['n']} tasks from `{shown}`). "
        "Made by `scripts/reward_audit.py`. Each number is a mean over the tasks.",
        "`correct`, `grounded` and the literal flag use the gated scorer. The flag is not part of the reward.",
        "",
        "| behaviour | note | gated reward | ungated reward | correct | grounded | answer is a SQL literal |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['behaviour']} | {r['note']} | {r['gated_reward']:.4f} | {r['ungated_reward']:.4f} | "
                     f"{r['correct_rate']:.3f} | {r['grounded_rate']:.3f} | {r['literal_flag_rate']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default=str(ROOT / "data" / "tasks" / "val.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "eval" / "reward_audit.md"))
    parser.add_argument("--db-dir", default=None)
    args = parser.parse_args(argv)
    tasks_path = Path(args.tasks).resolve()
    tasks = load_tasks(tasks_path)
    bad = sorted({t.split for t in tasks} - {"val"})
    if bad:
        print(f"ERROR: the audit uses the val split only, but {tasks_path} has splits {bad}", file=sys.stderr)
        return 2
    ensure_dbs(args.db_dir)
    text = render(audit(tasks, args.db_dir), tasks_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
