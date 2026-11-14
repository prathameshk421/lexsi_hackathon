import json

import pytest

from frugal_analyst.db import ensure_dbs
from frugal_analyst.policies import POLICIES, naive_policy, oracle_policy
from frugal_analyst.reward import ToolCall
from frugal_analyst.tasks import generate_tasks
from frugal_analyst.traces import build_trace, read_traces, run_scripted_policy, write_traces

ensure_dbs()

TOP_KEYS = [
    "task_id", "domain", "level", "split", "question", "gold_answer", "answer_type", "model", "steps",
    "predicted", "answered", "grounded", "supported", "correct", "answer_in_sql_literal", "efficiency", "reward",
    "cost", "optimal",
    "raw_completion", "completion_tokens", "latency_s",
]


@pytest.fixture(scope="module")
def splits():
    return generate_tasks(0)


def test_oracle_trace_has_every_field(splits):
    task = splits["test_id"][0]
    trace = build_trace(task, *oracle_policy(task), "oracle")
    assert list(trace) == TOP_KEYS
    assert trace["task_id"] == task.id and trace["split"] == "test_id" and trace["model"] == "oracle"
    assert trace["correct"] and trace["grounded"] and trace["answered"] and trace["supported"]
    assert trace["answer_in_sql_literal"] is False
    assert trace["efficiency"] == 1.0 and trace["reward"] == 1.0
    assert trace["predicted"] == task.gold_answer
    assert trace["optimal"] == {"tool_calls": task.optimal_tool_calls, "rows": task.optimal_rows, "vm_steps": task.optimal_vm_steps}
    assert trace["cost"]["tool_calls"] == task.optimal_tool_calls
    assert (trace["raw_completion"], trace["completion_tokens"], trace["latency_s"]) == (None, None, None)


def test_steps_have_the_documented_shape(splits):
    task = next(t for t in splits["test_id"] if t.level == 3)
    trace = build_trace(task, *oracle_policy(task), "oracle")
    *calls, final = trace["steps"]
    assert len(calls) == 3
    for step in calls:
        assert list(step) == ["type", "name", "arguments", "ok", "n_rows", "vm_steps", "output"]
        assert step["type"] == "tool_call" and step["ok"] is True
    assert calls[0]["name"] == "get_schema" and calls[0]["vm_steps"] == 0
    assert calls[2]["name"] == "run_sql" and calls[2]["n_rows"] == 1 and calls[2]["vm_steps"] > 0
    assert final == {"type": "final", "text": f"ANSWER: {task.gold_answer}"}
    assert trace["cost"]["vm_steps"] == sum(s["vm_steps"] for s in calls) == task.optimal_vm_steps


def test_optional_fields_are_passed_through(splits):
    task = splits["val"][0]
    calls, text = oracle_policy(task)
    trace = build_trace(task, calls, text, "zero-shot", raw_completion="raw", completion_tokens=17, latency_s=1.5)
    assert (trace["model"], trace["raw_completion"], trace["completion_tokens"], trace["latency_s"]) == ("zero-shot", "raw", 17, 1.5)


def test_trace_without_final_text(splits):
    task = splits["val"][0]
    trace = build_trace(task, [ToolCall("get_schema", {"database": task.domain})], None, "zero-shot")
    assert trace["steps"][-1]["type"] == "tool_call"
    assert not trace["answered"] and trace["predicted"] is None and trace["reward"] == 0.05


def test_failed_call_is_recorded(splits):
    task = splits["val"][0]
    trace = build_trace(task, [ToolCall("run_sql", {"database": task.domain, "query": "DROP TABLE x"})], "ANSWER: 1", "naive")
    step = trace["steps"][0]
    assert step["ok"] is False and step["output"].startswith("ERROR:") and step["vm_steps"] == 0
    assert trace["cost"]["failed_calls"] == 1


def test_literal_cheat_trace_is_not_supported(splits):
    task = splits["val"][0]
    query = f"SELECT '{task.gold_answer}' FROM {task.gold_sql.split(' FROM ')[1].split()[0]} LIMIT 1"
    trace = build_trace(task, [ToolCall("run_sql", {"database": task.domain, "query": query})],
                        f"ANSWER: {task.gold_answer}", "grpo")
    assert trace["grounded"] and not trace["supported"] and not trace["correct"] and trace["reward"] == 0.1


def test_old_trace_without_new_fields_is_still_readable_and_valid(tmp_path, splits):
    from frugal_analyst.metrics import aggregate, validate_trace

    trace = build_trace(splits["val"][0], *oracle_policy(splits["val"][0]), "oracle")
    del trace["supported"], trace["answer_in_sql_literal"]
    write_traces(tmp_path / "old.jsonl", [trace])
    (old,) = read_traces(tmp_path / "old.jsonl")
    assert validate_trace(old) == []
    agg = aggregate([old])
    assert agg["metrics"]["answer_in_sql_literal_rate"] is None and agg["counts"]["answer_in_sql_literal_rate"] == 0


def test_write_and_read_round_trip(tmp_path, splits):
    traces = [build_trace(t, *oracle_policy(t), "oracle") for t in splits["val"][:5]]
    path = tmp_path / "deep" / "t.jsonl"
    write_traces(path, traces)
    assert read_traces(path) == traces
    lines = path.read_text().splitlines()
    assert len(lines) == 5 and all(json.loads(line)["task_id"] for line in lines)


@pytest.mark.parametrize("name", ["oracle", "naive", "guess"])
def test_run_scripted_policy(tmp_path, splits, name):
    tasks = splits["test_id"][:20]
    out = tmp_path / f"{name}.jsonl"
    traces = run_scripted_policy(name, tasks, out)
    assert read_traces(out) == traces
    assert [t["task_id"] for t in traces] == [t.id for t in tasks]
    assert {t["model"] for t in traces} == {name}
    if name == "oracle":
        assert all(t["reward"] == 1.0 for t in traces)
    elif name == "naive":
        assert all(t["correct"] and t["efficiency"] < 1 for t in traces)
    else:
        assert all(t["reward"] == 0.0 and t["cost"]["tool_calls"] == 0 for t in traces)


def test_scripted_traces_are_byte_identical_across_runs(tmp_path, splits):
    tasks = splits["test_ood"][:15]
    run_scripted_policy("naive", tasks, tmp_path / "a.jsonl")
    run_scripted_policy("naive", tasks, tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_unknown_policy_name(tmp_path, splits):
    with pytest.raises(ValueError):
        run_scripted_policy("magic", splits["val"][:1], tmp_path / "x.jsonl")
    assert set(POLICIES) == {"oracle", "naive", "explorer", "guess"}


def test_explorer_policy_adds_one_preview_on_about_half_of_the_tasks(tmp_path, splits):
    from frugal_analyst.policies import explorer_policy

    tasks = splits["train"]
    extra = 0
    for task in tasks:
        calls, text = explorer_policy(task)
        oracle_calls, oracle_text = oracle_policy(task)
        assert text == oracle_text
        assert len(calls) - len(oracle_calls) in (0, 1) and len(calls) <= 4
        if len(calls) > len(oracle_calls):
            extra += 1
            assert calls[1].arguments["query"].endswith("LIMIT 5")
            assert calls[:1] + calls[2:] == oracle_calls  # the rest is the oracle path
    assert 0.4 < extra / len(tasks) < 0.6
    # Correct on every task, and never better than the oracle.
    traces = run_scripted_policy("explorer", tasks[:40], tmp_path / "e.jsonl")
    assert all(t["correct"] and 0.5 < t["reward"] <= 1.0 for t in traces)


def test_naive_policy_shape(splits):
    for task in splits["val"]:
        calls, text = naive_policy(task)
        assert calls[0].name == "get_schema" and calls[-1].arguments["query"] == task.gold_sql
        assert text == f"ANSWER: {task.gold_answer}"
        assert len(calls) > task.optimal_tool_calls
