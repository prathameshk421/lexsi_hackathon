"""Read AgentTune rollouts as tool calls plus a final answer, and score them.

AgentTune's tool loop (rollout_factory._execute_trajectory) gives us two views of
one episode:

1. A `Trajectory` object. Each step's `action["tool_calls"]` holds the calls the
   loop really executed, and `final_response` holds the last model turn. This is
   exact, so the GRPO reward and the runner use it when they have it.
2. A flat completion string. TRL decodes the rollout's `completion_ids` with
   `skip_special_tokens=True`. For a model whose chat template ignores `tools`
   (Tiny Aya, SmolLM2) the string is the model turns and the raw tool outputs
   glued together, with no separators:

       [{"tool_name": "get_schema", "parameters": {"database": "retail"}}]customers(...)
       [{"tool_name": "run_sql", "parameters": {...}}]n
       42ANSWER: 42

   `parse_completion` reads this string. It is the fallback when no trajectory
   is available, and it is what `raw_completion` in a trace file holds.

See docs/training-notes.md for the source lines behind this format.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .reward import RewardConfig, Score, ToolCall, score_trajectory
from .tasks import Task, load_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_DIR = REPO_ROOT / "data" / "tasks"

_ANSWER_PREFIX = "ANSWER:"


# ---------------------------------------------------------------------------
# Prompt and teacher-turn format (shared by SFT, GRPO and the runner)
# ---------------------------------------------------------------------------


def build_prompt_messages(task: Task) -> list[dict]:
    """The chat prompt for one task: system turn plus user turn.

    AgentTune adds the tool list to the system turn itself when the chat
    template ignores `tools` (see DEFAULT_TOOLS_FALLBACK_PROMPT in
    rollout_factory.py), so we do not add it here.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task)},
    ]


def format_tool_call_turn(calls: list[ToolCall]) -> str:
    """One assistant turn that calls tools, in the format AgentTune asks for.

    AgentTune's fallback prompt says: reply with only a JSON list,
    `[{"tool_name": "<name>", "parameters": {...}}]`. The teacher uses the same
    text so the student learns the format the rollout parser reads.
    """
    return json.dumps([{"tool_name": c.name, "parameters": c.arguments} for c in calls], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_helpers():
    """AgentTune's own parser helpers, imported late so this module loads fast."""
    from agenttune.agentic.rollout_engines import tool_call_parse as tcp

    return tcp


def completion_text(completion: Any) -> str:
    """The text of one TRL completion.

    TRL passes a plain string for string prompts and a list of message dicts
    (`[{"role": "assistant", "content": ...}]`) for chat prompts.
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, list):
        parts = [str(m.get("content") or "") for m in completion if isinstance(m, dict)]
        return "".join(parts) if parts else " ".join(str(m) for m in completion)
    return str(completion)


def _json_spans(text: str) -> list[tuple[int, int, Any]]:
    """Every JSON object or list in `text` with its start and end offsets.

    Same scan as AgentTune's tool_call_parse._json_objects, plus the offsets,
    because we need to know where the last call ends.
    """
    decoder = json.JSONDecoder()
    spans = []
    i = 0
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        spans.append((i, end, obj))
        i = max(end, i + 1)
    return spans


def _as_tool_calls(normalised: list[dict] | None) -> list[ToolCall]:
    """AgentTune's normalised call dicts as ToolCall objects."""
    calls = []
    for call in normalised or []:
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(fn.get("name") or "")
        calls.append(ToolCall(name, _arguments_dict(fn.get("arguments", {}))))
    return calls


def _arguments_dict(args: Any) -> dict:
    """Tool arguments as a dict. Bad arguments are kept so the call still counts as failed."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    # run_tool() reports a missing 'database' for this, so the call scores as failed.
    return {"_raw_arguments": args}


def _tool_call_batch(obj: Any, tcp) -> list[dict] | None:
    """The calls in one JSON value, or None if the value is not a tool call.

    Accepts what AgentTune's rollout accepts for these models: a Cohere-style
    list where every item has `tool_name` (tool_call_parse._parse_cohere_json),
    or dicts that look like `{"name": ..., "arguments": ...}`.
    """
    items = obj if isinstance(obj, list) else [obj]
    if not items or not all(isinstance(c, dict) for c in items):
        return None
    cohere = all("tool_name" in c or "tool_names" in c for c in items)
    generic = all(tcp._looks_like_tool_dict(c, require_args=True) for c in items)
    if not (cohere or generic):
        return None
    return tcp._as_calls(items)


def parse_completion(text: str) -> tuple[list[ToolCall], str | None]:
    """Tool calls and final text from AgentTune's flat completion string.

    Every JSON list or object shaped like a tool call counts as one batch of
    calls. The final text is what follows the last call. Tool output is glued
    to the next model turn, so if the rest holds `ANSWER:` we keep the text from
    the last `ANSWER:` on. Garbage input gives `([], text-or-None)` and never raises.
    """
    try:
        text = "" if text is None else str(text)
        tcp = _parse_helpers()
        cleaned = tcp._THINK_BLOCK_RE.sub("", text)
        calls: list[ToolCall] = []
        last_end = 0
        for start, end, obj in _json_spans(cleaned):
            batch = _tool_call_batch(obj, tcp)
            if batch:
                calls.extend(_as_tool_calls(batch))
                last_end = end
        rest = cleaned[last_end:]
        idx = rest.rfind(_ANSWER_PREFIX)
        final = rest[idx:].strip() if idx >= 0 else rest.strip()
        return calls, (final or None)
    except Exception:
        return [], None


def calls_from_trajectory(trajectory: Any) -> tuple[list[ToolCall], str | None]:
    """Executed tool calls and final text from an AgentTune Trajectory.

    Accepts the object or its `to_dict()` form (the trajectories.jsonl lines).
    In both, a tool step has `action = {"tool_calls": [...]}` and the terminal
    step has an empty action (rollout_factory._execute_trajectory).
    """
    if isinstance(trajectory, dict):
        steps = trajectory.get("steps") or []
        final = trajectory.get("final_response")
    else:
        steps = getattr(trajectory, "steps", None) or []
        final = getattr(trajectory, "final_response", None)
    calls: list[ToolCall] = []
    for step in steps:
        action = step.get("action") if isinstance(step, dict) else getattr(step, "action", None)
        raw_calls = (action or {}).get("tool_calls") if isinstance(action, dict) else None
        calls.extend(_as_tool_calls([c for c in raw_calls or [] if isinstance(c, dict)]))
    final = None if final is None else str(final)
    return calls, (final if final and final.strip() else None)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def load_task_index(tasks_dir: str | Path | None = None) -> dict[str, Task]:
    """Every task in `tasks_dir/*.jsonl`, keyed by id."""
    tasks_dir = Path(tasks_dir) if tasks_dir else DEFAULT_TASKS_DIR
    index: dict[str, Task] = {}
    for path in sorted(tasks_dir.glob("*.jsonl")):
        for task in load_tasks(path):
            index[task.id] = task
    if not index:
        raise FileNotFoundError(f"no tasks found in {tasks_dir}/*.jsonl. Run scripts/gen_data.py first.")
    return index


def score_completion(
    task: Task,
    completion: Any,
    trajectory: Any = None,
    mode: str = "gated",
    db_dir: str | Path | None = None,
) -> Score:
    """Score one rollout. Uses the trajectory when given, else parses the text."""
    if trajectory is not None:
        calls, final = calls_from_trajectory(trajectory)
    else:
        calls, final = parse_completion(completion_text(completion))
    return score_trajectory(task, calls, final, RewardConfig(mode=mode), db_dir)


def make_grpo_reward(
    mode: str = "gated",
    tasks_dir: str | Path | None = None,
    db_dir: str | Path | None = None,
) -> Callable[..., list[float]]:
    """A GRPO reward function for TRL/AgentTune.

    TRL calls it as `f(prompts=..., completions=..., completion_ids=..., **columns)`
    (GRPOTrainer._calculate_rewards). Dataset columns such as `task_id` arrive as
    lists aligned with `completions`. AgentTune's rollout adds `trajectories`.
    """
    if mode not in ("gated", "ungated"):
        raise ValueError(f"mode must be 'gated' or 'ungated', got {mode!r}")
    tasks = load_task_index(tasks_dir)

    def frugal_reward(completions, task_id=None, trajectories=None, **kwargs) -> list[float]:
        if task_id is None:
            raise ValueError("the GRPO dataset needs a 'task_id' column")
        unknown = sorted({t for t in task_id if t not in tasks})
        if unknown:
            raise KeyError(f"unknown task ids (regenerate data?): {unknown[:5]}")
        rewards, correct, efficiency, n_calls = [], [], [], []
        for i, completion in enumerate(completions):
            traj = trajectories[i] if isinstance(trajectories, list) and i < len(trajectories) else None
            try:
                score = score_completion(tasks[task_id[i]], completion, traj, mode, db_dir)
                rewards.append(float(score.reward))
                correct.append(float(score.correct))
                efficiency.append(float(score.efficiency))
                n_calls.append(float(score.cost.tool_calls))
            except Exception:
                # Model output must never crash training. Score it as nothing.
                rewards.append(0.0)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and correct:
            log_metric("frugal/correct", sum(correct) / len(correct))
            log_metric("frugal/efficiency", sum(efficiency) / len(efficiency))
            log_metric("frugal/tool_calls", sum(n_calls) / len(n_calls))
        return rewards

    return frugal_reward
