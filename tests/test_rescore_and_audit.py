"""Tests for scripts/rescore_traces.py and scripts/reward_audit.py. CPU only."""

import importlib.util
import json
from pathlib import Path

import pytest

from frugal_analyst.db import ensure_dbs
from frugal_analyst.policies import oracle_policy
from frugal_analyst.reward import ToolCall
from frugal_analyst.tasks import load_tasks
from frugal_analyst.traces import build_trace, read_traces, write_traces

ROOT = Path(__file__).resolve().parent.parent
ensure_dbs()


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rescore = load_script("rescore_traces")
audit = load_script("reward_audit")


@pytest.fixture(scope="module")
def val_tasks():
    return load_tasks(ROOT / "data" / "tasks" / "val.jsonl")


def old_style(trace, **fields):
    """A trace as the old reward wrote it: no evidence fields, and the given scores."""
    trace = dict(trace)
    del trace["supported"], trace["answer_in_sql_literal"]
    trace.update(fields)
    return trace


def test_rescore_finds_the_literal_cheat_and_keeps_honest_traces(tmp_path, val_tasks):
    honest = [build_trace(t, *oracle_policy(t), "grpo") for t in val_tasks[:3]]
    task = val_tasks[3]
    table = task.gold_sql.split(" FROM ")[1].split()[0]
    cheat_calls = [ToolCall("run_sql", {"database": task.domain, "query": f"SELECT '{task.gold_answer}' FROM {table} LIMIT 1"})]
    cheat = build_trace(task, cheat_calls, f"ANSWER: {task.gold_answer}", "grpo")
    # Pretend the old reward scored the cheat as correct with full marks.
    old = [old_style(t) for t in honest] + [old_style(cheat, correct=True, reward=1.0, efficiency=1.0)]
    src = tmp_path / "in" / "grpo_val.jsonl"
    write_traces(src, old)
    out = tmp_path / "out"
    assert rescore.main(["--traces", str(src), "--out", str(out)]) == 0
    new = read_traces(out / "grpo_val.jsonl")
    assert [t["correct"] for t in new] == [True, True, True, False]
    assert new[3]["reward"] == 0.1 and new[3]["supported"] is False
    summary = json.loads((out / "rescore_summary.json").read_text())[0]
    assert summary["correct_changed"] == 1 and summary["correct_true_to_false"] == 1 and summary["reward_changed"] == 1
    assert summary["grounded_changed"] == 0
    assert read_traces(src) == old  # the input file is unchanged


def test_rescore_fails_loudly_when_a_replayed_output_differs(tmp_path, val_tasks, capsys):
    trace = old_style(build_trace(val_tasks[0], *oracle_policy(val_tasks[0]), "grpo"))
    trace["steps"][1]["output"] = "COUNT(*)\n999999"
    src = write_and_path(tmp_path, [trace])
    out = tmp_path / "out"
    assert rescore.main(["--traces", str(src), "--out", str(out)]) == rescore.EXIT_REPLAY_MISMATCH
    assert "differ from the recorded outputs" in capsys.readouterr().err
    assert not out.exists()


def test_rescore_vm_drift_needs_the_flag(tmp_path, val_tasks):
    trace = old_style(build_trace(val_tasks[0], *oracle_policy(val_tasks[0]), "grpo"))
    trace["steps"][1]["vm_steps"] += 10
    src = write_and_path(tmp_path, [trace])
    assert rescore.main(["--traces", str(src), "--out", str(tmp_path / "a")]) == rescore.EXIT_REPLAY_MISMATCH
    assert rescore.main(["--traces", str(src), "--out", str(tmp_path / "b"), "--allow-vm-drift"]) == 0
    summary = json.loads((tmp_path / "b" / "rescore_summary.json").read_text())[0]
    assert summary["vm_drift_traces"] == 1


def test_rescore_refuses_to_write_into_the_input_folder(tmp_path, val_tasks):
    src = write_and_path(tmp_path, [build_trace(val_tasks[0], *oracle_policy(val_tasks[0]), "grpo")])
    assert rescore.main(["--traces", str(src), "--out", str(src.parent)]) == rescore.EXIT_BAD_INPUT


def write_and_path(tmp_path, traces):
    src = tmp_path / "in" / "t.jsonl"
    write_traces(src, traces)
    return src


def test_reward_audit_table(tmp_path, val_tasks):
    rows = {r["behaviour"]: r for r in audit.audit(val_tasks)}
    assert rows["oracle"]["gated_reward"] == 1.0
    assert rows["naive"]["gated_reward"] < rows["oracle + one LIMIT 5 preview"]["gated_reward"] < 1.0
    for cheat in ("SELECT 1, then gold answer", "SELECT '<gold>' literal", "gold answer, no query"):
        assert rows[cheat]["gated_reward"] <= 0.05, cheat
        assert rows[cheat]["ungated_reward"] == 0.5, cheat  # the ungated reward pays for it: the hack
    assert rows["SELECT '<gold>' FROM t LIMIT 1"]["gated_reward"] == 0.1
    assert rows["wrong answer after SELECT 1"]["gated_reward"] == 0.05
    assert rows["honest wrong attempt"]["gated_reward"] == 0.1
    for cheat in ("literal hidden in a CTE", "literal in a FROM subquery", "literal in VALUES",
                  "literal in a scalar subquery", "literal as a CASE result"):
        assert rows[cheat]["gated_reward"] <= 0.1 and rows[cheat]["correct_rate"] == 0.0, cheat
    # Known gap, kept visible on purpose: an answer computed from other literals still gets near full reward.
    assert rows["answer built by arithmetic or concatenation"]["gated_reward"] > 0.9
    out = tmp_path / "audit.md"
    assert audit.main(["--out", str(out)]) == 0
    assert "| oracle |" in out.read_text() and chr(0x2014) not in out.read_text()


def test_reward_audit_refuses_test_splits(tmp_path):
    code = audit.main(["--tasks", str(ROOT / "data" / "tasks" / "test_id.jsonl"), "--out", str(tmp_path / "x.md")])
    assert code == 2 and not (tmp_path / "x.md").exists()
