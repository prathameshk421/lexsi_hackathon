"""Trace records: the only interface between runs, evaluation and the demo.

One JSON object per line. See docs/spec.md for the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from .env import run_tool
from .policies import POLICIES
from .reward import RewardConfig, ToolCall, score_trajectory
from .tasks import Task


def build_trace(
    task: Task,
    calls: list[ToolCall],
    final_text: str | None,
    model_label: str,
    raw_completion: str | None = None,
    completion_tokens: int | None = None,
    latency_s: float | None = None,
    db_dir: str | Path | None = None,
    cfg: RewardConfig = RewardConfig(),
) -> dict:
    """Run the calls, score them and return a full trace record."""
    steps = []
    for call in calls:
        result = run_tool(call.name, call.arguments, db_dir)
        steps.append({
            "type": "tool_call", "name": call.name, "arguments": call.arguments,
            "ok": result.ok, "n_rows": result.n_rows, "vm_steps": result.vm_steps, "output": result.output,
        })
    if final_text is not None:
        steps.append({"type": "final", "text": final_text})
    score = score_trajectory(task, calls, final_text, cfg, db_dir)
    return {
        "task_id": task.id, "domain": task.domain, "level": task.level, "split": task.split,
        "question": task.question, "gold_answer": task.gold_answer, "answer_type": task.answer_type,
        "model": model_label,
        "steps": steps,
        "predicted": score.predicted, "answered": score.answered, "grounded": score.grounded, "correct": score.correct,
        "efficiency": score.efficiency, "reward": score.reward,
        "cost": {
            "tool_calls": score.cost.tool_calls, "sql_calls": score.cost.sql_calls, "failed_calls": score.cost.failed_calls,
            "rows_returned": score.cost.rows_returned, "vm_steps": score.cost.vm_steps,
        },
        "optimal": {"tool_calls": task.optimal_tool_calls, "rows": task.optimal_rows, "vm_steps": task.optimal_vm_steps},
        "raw_completion": raw_completion, "completion_tokens": completion_tokens, "latency_s": latency_s,
    }


def write_traces(path: str | Path, traces: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def read_traces(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_scripted_policy(
    policy_name: str, tasks: list[Task], out_path: str | Path, db_dir: str | Path | None = None
) -> list[dict]:
    """Run `oracle`, `naive` or `guess` over tasks, write the traces, return them."""
    if policy_name not in POLICIES:
        raise ValueError(f"unknown policy {policy_name!r}. Valid: {', '.join(POLICIES)}")
    policy = POLICIES[policy_name]
    traces = []
    for task in tasks:
        calls, final_text = policy(task)
        traces.append(build_trace(task, calls, final_text, policy_name, db_dir=db_dir))
    write_traces(out_path, traces)
    return traces
