"""Tests for frugal_analyst.trajectory.

The flat strings below follow the real format AgentTune hands to a GRPO reward
function for Tiny Aya: each model turn (a JSON call list), then the raw tool
output, glued with no separator, then the final turn. See docs/training-notes.md.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("agenttune")

from frugal_analyst.db import ensure_dbs
from frugal_analyst.env import TOOLS, run_tool
from frugal_analyst.reward import ToolCall
from frugal_analyst.trajectory import (
    DEFAULT_TASKS_DIR,
    build_prompt_messages,
    calls_from_trajectory,
    completion_text,
    format_tool_call_turn,
    load_task_index,
    make_grpo_reward,
    parse_completion,
)


@pytest.fixture(scope="module")
def tasks():
    if not (DEFAULT_TASKS_DIR / "train.jsonl").exists():
        pytest.skip("data/tasks missing: run scripts/gen_data.py")
    ensure_dbs()
    return load_task_index()


@pytest.fixture(scope="module")
def task(tasks):
    """A level 3 train task: schema, code lookup, final query."""
    return next(t for t in tasks.values() if t.split == "train" and t.level == 3)


def oracle_turns(task) -> tuple[list[str], list[str]]:
    """The oracle's model turns and the tool output after each call turn."""
    turns, outputs = [], []
    for call in task.oracle_calls:
        tc = ToolCall(call["name"], call["arguments"])
        turns.append(format_tool_call_turn([tc]))
        outputs.append(run_tool(tc.name, tc.arguments).output)
    turns.append(f"ANSWER: {task.gold_answer}")
    return turns, outputs


def flat_completion(turns: list[str], outputs: list[str]) -> str:
    """Glue turns and tool outputs the way TRL decodes AgentTune's completion_ids."""
    text = ""
    for i, turn in enumerate(turns):
        text += turn
        if i < len(outputs):
            text += outputs[i]
    return text


# ---------------------------------------------------------------------------
# parse_completion
# ---------------------------------------------------------------------------


def test_parse_oracle_flat_completion(task):
    turns, outputs = oracle_turns(task)
    calls, final = parse_completion(flat_completion(turns, outputs))
    assert [c.name for c in calls] == [c["name"] for c in task.oracle_calls]
    assert [c.arguments for c in calls] == [c["arguments"] for c in task.oracle_calls]
    assert final == f"ANSWER: {task.gold_answer}"


def test_answer_glued_to_tool_output():
    text = (
        '[{"tool_name": "run_sql", "parameters": {"database": "retail", "query": "SELECT COUNT(*) FROM orders"}}]'
        "COUNT(*)\n1890"
        "The table has 1890 orders.\nANSWER: 1890"
    )
    calls, final = parse_completion(text)
    assert len(calls) == 1 and calls[0].name == "run_sql"
    assert calls[0].arguments == {"database": "retail", "query": "SELECT COUNT(*) FROM orders"}
    assert final == "ANSWER: 1890"


def test_two_calls_in_one_turn():
    text = (
        '[{"tool_name": "get_schema", "parameters": {"database": "school"}}, '
        '{"tool_name": "run_sql", "parameters": {"database": "school", "query": "SELECT 1"}}]'
        "students(...)1\n1ANSWER: 1"
    )
    calls, final = parse_completion(text)
    assert [c.name for c in calls] == ["get_schema", "run_sql"]
    assert final == "ANSWER: 1"


def test_generic_name_arguments_shape_is_parsed():
    # AgentTune also accepts {"name", "arguments"} (Qwen style <tool_call> tags).
    text = '<tool_call>{"name": "get_schema", "arguments": {"database": "saas"}}</tool_call>accounts(...)ANSWER: 3'
    calls, final = parse_completion(text)
    assert [(c.name, c.arguments) for c in calls] == [("get_schema", {"database": "saas"})]
    assert final == "ANSWER: 3"


def test_matches_agenttune_rollout_parser(task):
    """Each turn parses to the same call in our parser and in the rollout's own parser."""
    from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

    turns, _ = oracle_turns(task)
    for turn in turns[:-1]:
        ours, _ = parse_completion(turn)
        theirs = _extract_tool_calls({"role": "assistant", "content": turn}, turn)
        assert [(c.name, c.arguments) for c in ours] == [
            (t["function"]["name"], json.loads(t["function"]["arguments"])) for t in theirs
        ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "I don't know.",
        "[2, 3]",
        '{"answer": 42}',
        '[{"tool_name": "run_sql", "parameters": {"database": "clinic", "query": "SELECT',  # cut mid-call
        '[{"tool_name": "run_sql"' * 50,
        "\x00\x01�]]]}}}{{{[[[",
        "ANSWER:",
    ],
)
def test_garbage_never_raises_and_finds_no_call(text):
    calls, final = parse_completion(text)
    assert calls == []
    assert final is None or isinstance(final, str)


def test_bad_arguments_kept_as_failed_call():
    calls, _ = parse_completion('[{"tool_name": "run_sql", "parameters": "SELECT 1"}]')
    assert len(calls) == 1 and calls[0].name == "run_sql"
    assert "query" not in calls[0].arguments  # run_tool() will report it as failed


def test_completion_text_shapes():
    assert completion_text("abc") == "abc"
    assert completion_text([{"role": "assistant", "content": "abc"}]) == "abc"
    assert completion_text({"role": "assistant", "content": "abc"}) == "abc"
    assert completion_text(None) == ""


def test_format_tool_call_turn_round_trip():
    call = ToolCall("run_sql", {"database": "retail", "query": "SELECT 'a' AS x"})
    calls, final = parse_completion(format_tool_call_turn([call]))
    assert calls == [call] and final is None


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def test_reward_oracle_flat_text_scores_one(task):
    reward = make_grpo_reward("gated")
    turns, outputs = oracle_turns(task)
    as_text = flat_completion(turns, outputs)
    as_chat = [{"role": "assistant", "content": as_text}]
    assert reward([as_text, as_chat], task_id=[task.id, task.id]) == [1.0, 1.0]


def test_reward_garbage_is_zero(task):
    reward = make_grpo_reward("gated")
    junk = ["", None, "[2, 3]", '[{"tool_name": "drop_table", "parameters": {}}]', 12345, {"weird": True}]
    assert reward(junk, task_id=[task.id] * len(junk)) == [0.0] * len(junk)


def test_reward_guess_without_query_is_not_grounded(task):
    guess = f"ANSWER: {task.gold_answer}"
    assert make_grpo_reward("gated")([guess], task_id=[task.id]) == [0.0]
    # The ungated ablation reward pays for the free answer: this is the hack.
    assert make_grpo_reward("ungated")([guess], task_id=[task.id])[0] > 0.0


def test_reward_wasteful_but_correct_is_below_oracle(task):
    turns, outputs = oracle_turns(task)
    dump = ToolCall("run_sql", {"database": task.domain, "query": "SELECT * FROM " + task.gold_sql.split("FROM")[1].split()[0]})
    wasteful = format_tool_call_turn([dump]) + run_tool(dump.name, dump.arguments).output
    reward = make_grpo_reward("gated")
    oracle_r, wasteful_r = reward([flat_completion(turns, outputs), wasteful + flat_completion(turns, outputs)], task_id=[task.id] * 2)
    assert oracle_r == 1.0 and 0.5 <= wasteful_r < oracle_r


def test_reward_needs_task_ids(task):
    reward = make_grpo_reward("gated")
    with pytest.raises(ValueError):
        reward(["x"])
    with pytest.raises(KeyError):
        reward(["x"], task_id=["no-such-task"])


def test_logs_metrics_when_trl_passes_log_metric(task):
    seen = {}
    make_grpo_reward("gated")(["x"], task_id=[task.id], log_metric=lambda k, v: seen.__setitem__(k, v))
    assert set(seen) == {"frugal/correct", "frugal/efficiency", "frugal/tool_calls"}


# ---------------------------------------------------------------------------
# Real AgentTune rollout loop (no model: a scripted engine replays the oracle)
# ---------------------------------------------------------------------------


def run_scripted_rollout(task, turns, max_steps=5):
    from agenttune.agentic.rollout_engines.base import RolloutEngine
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    class Replay(RolloutEngine):
        def _get_tokenizer(self):
            return None

        def generate(self, prompts, tools, gen_cfg):
            turn = sum(1 for m in prompts if m.get("role") == "assistant")
            text = turns[turn] if turn < len(turns) else ""
            return {"completions": [text], "logprobs": [None], "metadata": {}}

    rollout = create_rollout_fn(rollout_engine=Replay(), tools=list(TOOLS), max_steps=max_steps)
    return rollout([build_prompt_messages(task)])


def test_max_steps_k_runs_k_calls_and_still_answers(task):
    # AgentTune's max_steps counts executed tool rounds. After the last round it
    # always generates one more turn, so k calls need max_steps = k, not k + 1.
    turns, _ = oracle_turns(task)
    k = len(task.oracle_calls)
    calls, final = calls_from_trajectory(run_scripted_rollout(task, turns, max_steps=k)["trajectories"][0])
    assert len(calls) == k and final == f"ANSWER: {task.gold_answer}"
    # One step fewer: the k-th call is written but not run, and there is no answer.
    calls, final = calls_from_trajectory(run_scripted_rollout(task, turns, max_steps=k - 1)["trajectories"][0])
    assert len(calls) == k - 1 and not (final or "").startswith("ANSWER:")


def test_trajectory_from_real_rollout_loop(task):
    turns, _ = oracle_turns(task)
    out = run_scripted_rollout(task, turns)
    traj = out["trajectories"][0]
    calls, final = calls_from_trajectory(traj)
    assert [(c.name, c.arguments) for c in calls] == [(c["name"], c["arguments"]) for c in task.oracle_calls]
    assert final == f"ANSWER: {task.gold_answer}"
    # The to_dict() form written to trajectories.jsonl parses the same way.
    assert calls_from_trajectory(json.loads(json.dumps(traj.to_dict(), default=str))) == (calls, final)
    # The reward prefers the trajectory over the text.
    reward = make_grpo_reward("gated")
    assert reward(["text is ignored"], task_id=[task.id], trajectories=[traj]) == [1.0]


def test_trajectory_with_unknown_tool_scores_low(task):
    turns = ['[{"tool_name": "delete_everything", "parameters": {}}]', f"ANSWER: {task.gold_answer}"]
    out = run_scripted_rollout(task, turns)
    calls, final = calls_from_trajectory(out["trajectories"][0])
    assert [c.name for c in calls] == ["delete_everything"]
    assert make_grpo_reward("gated")(["x"], task_id=[task.id], trajectories=out["trajectories"]) == [0.0]
