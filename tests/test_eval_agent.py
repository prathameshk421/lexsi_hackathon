"""Tests for scripts/eval_agent.py. They build small trace files from real tasks and run the real script."""

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("eval_agent", ROOT / "scripts" / "eval_agent.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

from frugal_analyst import env  # noqa: E402
from frugal_analyst.policies import oracle_policy  # noqa: E402
from frugal_analyst.reward import ToolCall  # noqa: E402
from frugal_analyst.tasks import load_tasks  # noqa: E402
from frugal_analyst.traces import build_trace, run_scripted_policy, write_traces  # noqa: E402


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(ROOT / "data" / "tasks" / "test_id.jsonl")[:8]


@pytest.fixture(scope="module")
def files(tasks, tmp_path_factory):
    d = tmp_path_factory.mktemp("agent_traces")
    out = {name: d / f"{name}_test_id.jsonl" for name in ("oracle", "naive", "guess")}
    for name, path in out.items():
        run_scripted_policy(name, tasks, path)
    return out


def run_script(tmp_path, paths, *extra):
    out = tmp_path / "out"
    code = script.main(["--traces", *[str(p) for p in paths], "--out", str(out), "--bootstrap", "200", *extra])
    return code, out


def load(out):
    return json.loads((out / "agent_results.json").read_text())


def model_row(out, label, split="test_id"):
    return load(out)["splits"][split]["models"][label]


def test_oracle_scores_one_everywhere(tmp_path, files):
    code, out = run_script(tmp_path, [files["oracle"]], "--baseline", "oracle")
    assert code == 0
    m = model_row(out, "oracle")
    assert m["n"] == 8 and m["failed_episodes"] == 0
    for key in ("tool_call_precision_name", "tool_call_recall_name", "tool_call_f1_name", "tool_call_f1",
                "task_success", "agent_loop_detection", "tool_call_validity"):
        assert m["metrics"][key] == pytest.approx(1.0), key
    assert m["metrics"]["redundant_tool_calls"] == 0.0
    assert m["counts"]["redundant_tool_calls"] == 8


def test_naive_has_lower_precision_than_oracle(tmp_path, files):
    code, out = run_script(tmp_path, [files["oracle"], files["naive"]], "--baseline", "naive")
    assert code == 0
    naive, oracle = model_row(out, "naive"), model_row(out, "oracle")
    assert naive["metrics"]["tool_call_precision_name"] < oracle["metrics"]["tool_call_precision_name"]
    assert naive["metrics"]["tool_call_recall_name"] == pytest.approx(1.0)  # it still makes every oracle call
    assert naive["metrics"]["task_success"] == 1.0
    pair = load(out)["comparisons"]["test_id"]["pairs"][0]
    assert pair["model"] == "oracle"
    f1 = pair["metrics"]["tool_call_f1_name"]
    assert f1["delta"] > 0 and f1["ci_low"] <= f1["delta"] <= f1["ci_high"] and f1["verdict"] == "better"
    md = (out / "agent_results.md").read_text()
    assert "(headline)" in md and chr(0x2014) not in md  # no em dashes
    assert (out / "agent_runs" / "naive_test_id.json").is_file()


def test_repeated_failing_call_is_caught(tmp_path, tasks):
    task = tasks[0]
    calls, final = oracle_policy(task)
    bad = ToolCall("run_sql", {"database": task.domain, "query": "SELECT * FROM no_such_table"})
    trace = build_trace(task, calls + [bad, bad, bad], final, "loopy")
    path = tmp_path / "loopy_test_id.jsonl"
    write_traces(path, [trace])
    code, out = run_script(tmp_path, [path], "--baseline", "loopy")
    assert code == 0
    m = model_row(out, "loopy")["metrics"]
    assert m["agent_loop_detection"] == 0.0  # the same call 3 times is a loop
    assert m["redundant_tool_calls"] == pytest.approx(2 / 5)  # 2 repeats among 5 calls
    assert m["tool_call_recall_name"] == 1.0
    assert m["tool_call_precision_name"] == pytest.approx(2 / 5)  # the 3 bad calls are extra
    assert m["tool_call_f1"] == pytest.approx(2 * 0.4 * 1.0 / 1.4)  # the oracle calls still match exactly
    assert model_row(out, "loopy")["crosscheck"]["ok"]


def test_trace_with_no_calls_scores_zero_without_crashing(tmp_path, files):
    code, out = run_script(tmp_path, [files["guess"], files["oracle"]], "--baseline", "guess")
    assert code == 0  # an episode with no calls is not a failed episode
    g = model_row(out, "guess")
    assert g["failed_episodes"] == 0
    for key in ("tool_call_precision_name", "tool_call_recall_name", "tool_call_f1_name", "tool_call_f1"):
        assert g["metrics"][key] == 0.0 and g["counts"][key] == 8
    for key in ("redundant_tool_calls", "agent_loop_detection", "tool_call_validity"):
        assert g["metrics"][key] is None and g["counts"][key] == 0  # nothing to measure, said so
    assert g["crosscheck"]["ok"]
    assert "n/a" in (out / "agent_results.md").read_text()


def test_crosscheck_passes_and_catches_a_difference(tmp_path, files, monkeypatch):
    code, out = run_script(tmp_path, [files["oracle"], files["naive"], files["guess"]], "--baseline", "naive")
    assert code == 0
    res = load(out)
    assert res["crosscheck_ok"] is True
    assert all(m["crosscheck"]["checked_episodes"] == 8 for m in res["splits"]["test_id"]["models"].values())

    real = script.multiset_prf
    monkeypatch.setattr(script, "multiset_prf", lambda p, r, a: {**real(p, r, a), "f1": 0.5})
    code, out = run_script(tmp_path, [files["oracle"]], "--baseline", "oracle")
    assert code == script.EXIT_MISMATCH
    assert load(out)["crosscheck_ok"] is False  # still written, so the difference is visible


def test_multiset_counts_a_repeat_as_a_false_positive():
    get = {"name": "get_schema", "arguments": {"database": "retail"}}
    sql = {"name": "run_sql", "arguments": {"database": "retail", "query": "SELECT 1"}}
    p = script.multiset_prf([get, sql, sql], [get, sql], with_args=True)
    assert p["precision"] == pytest.approx(2 / 3) and p["recall"] == 1.0
    other = {"name": "run_sql", "arguments": {"database": "retail", "query": "SELECT 2"}}
    assert script.multiset_prf([get, other], [get, sql], with_args=True)["f1"] == pytest.approx(0.5)
    assert script.multiset_prf([get, other], [get, sql], with_args=False)["f1"] == 1.0  # names only


def test_failed_episode_gives_exit_code_1(tmp_path, files, monkeypatch):
    def boom(episode, case):
        raise ValueError("outcome bug")

    monkeypatch.setattr(script, "is_correct", boom)
    code, out = run_script(tmp_path, [files["oracle"]], "--baseline", "oracle")
    assert code == script.EXIT_FAILED_EPISODES
    m = model_row(out, "oracle")
    assert m["failed_episodes"] == 8 and m["failures"][0]["status"] == "method_error"


def test_bad_trace_file_gives_exit_code_2(tmp_path, files, capsys):
    bad = tmp_path / "bad_test_id.jsonl"
    bad.write_text(files["oracle"].read_text().splitlines()[0] + "\n{not json\n", encoding="utf-8")
    code, out = run_script(tmp_path, [bad], "--baseline", "oracle")
    assert code == script.EXIT_BAD_INPUT
    assert "bad_test_id.jsonl:2" in capsys.readouterr().err
    assert not (out / "agent_results.json").exists()


def test_unknown_task_id_gives_exit_code_2(tmp_path, files, capsys):
    rec = json.loads(files["oracle"].read_text().splitlines()[0])
    rec["task_id"] = "retail-L1-9999"
    path = tmp_path / "oracle_test_id.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    code, _ = run_script(tmp_path, [path], "--baseline", "oracle")
    assert code == script.EXIT_BAD_INPUT
    assert "no oracle calls" in capsys.readouterr().err


def test_missing_task_file_and_unknown_baseline_give_exit_code_2(tmp_path, files, capsys):
    code, _ = run_script(tmp_path, [files["oracle"]], "--baseline", "oracle", "--tasks-dir", str(tmp_path / "nope"))
    assert code == script.EXIT_BAD_INPUT and "no task file" in capsys.readouterr().err
    code, _ = run_script(tmp_path, [files["oracle"]], "--baseline", "nope")
    assert code == script.EXIT_BAD_INPUT and "nope" in capsys.readouterr().err


def test_run_json_files_are_skipped(tmp_path, files):
    run_json = tmp_path / "oracle_test_id.run.json"
    run_json.write_text("{}", encoding="utf-8")
    code, out = run_script(tmp_path, [files["oracle"], run_json], "--baseline", "oracle")
    assert code == 0
    assert [i["path"] for i in load(out)["inputs"]] == [str(files["oracle"])]


def test_template_group_breakdown_uses_known_groups(tmp_path, files):
    code, out = run_script(tmp_path, [files["oracle"]], "--baseline", "oracle")
    assert code == 0
    groups = model_row(out, "oracle")["by_template_group"]
    assert set(groups) <= {"rl_only", "sft_seen"} and sum(g["n"] for g in groups.values()) == 8


def test_tool_schemas_match_the_real_tools():
    declared = {s["function"]["name"]: s["function"]["parameters"] for s in script.TOOL_SCHEMAS}
    assert set(declared) == {f.__name__ for f in env.TOOLS}
    for f in env.TOOLS:
        params = list(inspect.signature(f).parameters)
        assert list(declared[f.__name__]["properties"]) == params
        assert declared[f.__name__]["required"] == params


def test_runs_on_the_real_traces_if_present(tmp_path):
    traces = sorted((ROOT / "outputs" / "traces").glob("*_test_*.jsonl")) if (ROOT / "outputs" / "traces").exists() else []
    if not any(p.name.startswith("oracle") for p in traces):
        pytest.skip("scripted trace files not generated")
    code, out = run_script(tmp_path, traces, "--baseline", "oracle")
    assert code == 0
    for split in load(out)["splits"].values():
        oracle = split["models"]["oracle"]["metrics"]
        assert oracle["tool_call_f1_name"] == 1.0 and oracle["tool_call_f1"] == 1.0
