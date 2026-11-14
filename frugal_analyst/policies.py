"""Scripted policies. No model and no GPU.

Each policy returns `(calls, final_text)`. They are used for teacher data,
for reward tests, and as baselines in the demo.
"""

from __future__ import annotations

import re
import zlib

from .reward import ToolCall
from .tasks import Task


def oracle_policy(task: Task) -> tuple[list[ToolCall], str]:
    """The efficient path: schema, optional lookup, one final query."""
    calls = [ToolCall(c["name"], dict(c["arguments"])) for c in task.oracle_calls]
    return calls, f"ANSWER: {task.gold_answer}"


def _tables_in(sql: str) -> list[str]:
    """Table names after FROM or JOIN, in order, without repeats."""
    seen: list[str] = []
    for name in re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", sql, flags=re.IGNORECASE):
        if name not in seen:
            seen.append(name)
    return seen


def naive_policy(task: Task) -> tuple[list[ToolCall], str]:
    """Correct but wasteful: dump every table, sometimes slip once, then answer.

    It models an untrained agent that gets the right answer at a high cost.
    """
    db = task.domain
    tables = _tables_in(task.gold_sql)
    calls = [ToolCall("get_schema", {"database": db})]
    calls += [ToolCall("run_sql", {"database": db, "query": f"SELECT * FROM {t}"}) for t in tables]
    if zlib.crc32(task.id.encode()) % 2 == 0:  # a stable "maybe": half of the tasks
        calls.append(ToolCall("run_sql", {"database": db, "query": f"SELECT COUNT(*) FROM {tables[0]} WHERE no_such_column = 1"}))
    if task.level == 3:
        calls.append(ToolCall("run_sql", dict(task.oracle_calls[1]["arguments"])))  # the code lookup
    calls.append(ToolCall("run_sql", {"database": db, "query": task.gold_sql}))
    return calls, f"ANSWER: {task.gold_answer}"


def explorer_policy(task: Task) -> tuple[list[ToolCall], str]:
    """The oracle path, plus a 5-row preview of the first table on half of the tasks.

    It is the "mixed teacher" experiment: the preview is correct but not needed.
    SFT on these traces shows the model both habits. GRPO must then learn to
    drop the preview. The write-up must say that we added this waste on purpose.
    """
    calls, final = oracle_policy(task)
    if zlib.crc32(("preview:" + task.id).encode()) % 2 == 0:  # a stable half of the tasks
        table = _tables_in(task.gold_sql)[0]
        preview = ToolCall("run_sql", {"database": task.domain, "query": f"SELECT * FROM {table} LIMIT 5"})
        calls.insert(1, preview)  # right after get_schema
    return calls, final


def guess_policy(task: Task) -> tuple[list[ToolCall], str]:
    """No tool calls, a made-up answer."""
    return [], "ANSWER: 42" if task.answer_type == "number" else "ANSWER: unknown"


POLICIES = {"oracle": oracle_policy, "naive": naive_policy, "explorer": explorer_policy, "guess": guess_policy}
