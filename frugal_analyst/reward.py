"""Reward: correctness gates the efficiency bonus.

`score_trajectory` runs every tool call again against the databases. It never
trusts the tool output text a model saw. This keeps the reward stateless and
deterministic, and it means a model cannot fake a tool result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .env import run_tool
from .tasks import Task


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class EpisodeCost:
    tool_calls: int
    sql_calls: int
    failed_calls: int
    rows_returned: int
    vm_steps: int


@dataclass
class Score:
    answered: bool
    grounded: bool
    correct: bool
    efficiency: float  # 0..1, defined even when incorrect (for analysis)
    reward: float
    cost: EpisodeCost
    predicted: str | None


@dataclass
class RewardConfig:
    # "gated": efficiency only counts when the answer is correct.
    # "ungated": the hackable reward for the ablation. A model that never
    #            queries gets full efficiency marks.
    mode: str = "gated"


# ---------------------------------------------------------------------------
# Answer handling
# ---------------------------------------------------------------------------

_ANSWER_PREFIX = "ANSWER:"


def extract_answer(text: str | None) -> str | None:
    """The value after `ANSWER:` on the last non-empty line, or None."""
    if not text:
        return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or not lines[-1].startswith(_ANSWER_PREFIX):
        return None
    value = lines[-1][len(_ANSWER_PREFIX):].strip()
    return value or None


def _parse_number(text: str) -> float | None:
    cleaned = re.sub(r"[,$€£₹\s]", "", text)
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _clean_text(text: str) -> str:
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text.casefold()


def answers_match(predicted: str, gold: str, answer_type: str) -> bool:
    if answer_type == "number":
        a, b = _parse_number(predicted), _parse_number(gold)
        if a is None or b is None:
            return False
        diff = abs(a - b)
        return diff <= 0.01 or diff <= 0.001 * abs(b)
    return _clean_text(predicted) == _clean_text(gold)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_trajectory(
    task: Task,
    calls: list[ToolCall],
    final_text: str | None,
    cfg: RewardConfig = RewardConfig(),
    db_dir: str | Path | None = None,
) -> Score:
    sql_calls = failed = rows = vm_steps = 0
    any_valid_call = grounded = False
    for call in calls:
        result = run_tool(call.name, call.arguments, db_dir)
        if not result.ok:
            failed += 1
        else:
            any_valid_call = True
        if result.is_sql:
            sql_calls += 1
            vm_steps += result.vm_steps
            if result.ok:
                rows += result.n_rows
                grounded = True
    cost = EpisodeCost(len(calls), sql_calls, failed, rows, vm_steps)

    predicted = extract_answer(final_text)
    answered = predicted is not None
    correct = answered and grounded and answers_match(predicted, task.gold_answer, task.answer_type)

    if cost.tool_calls == 0:
        efficiency = 0.0
    else:
        efficiency = (
            min(1.0, task.optimal_tool_calls / cost.tool_calls)
            + min(1.0, (task.optimal_rows + 1) / (cost.rows_returned + 1))
            + min(1.0, (task.optimal_vm_steps + 1) / (cost.vm_steps + 1))
        ) / 3

    if cfg.mode == "ungated":
        efficiency_naive = 1.0 if cost.tool_calls == 0 else efficiency  # zero calls looks perfectly cheap
        reward = 0.5 * correct + 0.5 * efficiency_naive
    elif correct:
        reward = 0.5 + 0.5 * efficiency
    elif grounded and answered:
        reward = 0.1
    elif any_valid_call:
        reward = 0.05
    else:
        reward = 0.0
    return Score(answered, grounded, correct, efficiency, float(reward), cost, predicted)
