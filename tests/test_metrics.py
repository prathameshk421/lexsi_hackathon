"""Tests for frugal_analyst.metrics. Expected numbers are worked out by hand."""

import copy

import pytest

from frugal_analyst import metrics as M


def make_trace(task_id="t1", model="m", split="test_id", level=1, domain="retail", correct=True, answered=True,
               grounded=True, tool_calls=2, failed=0, rows=1, vm=100, optimal=(2, 1, 100), efficiency=1.0,
               reward=1.0, latency=None, tokens=None, final=True):
    steps = [{"type": "tool_call", "name": "run_sql", "arguments": {"database": domain, "query": "SELECT 1"},
              "ok": i >= failed, "n_rows": 1, "vm_steps": 10, "output": "x"} for i in range(tool_calls)]
    if final:
        steps.append({"type": "final", "text": "ANSWER: 1"})
    return {
        "task_id": task_id, "domain": domain, "level": level, "split": split, "question": "q?", "gold_answer": "1",
        "answer_type": "number", "model": model, "steps": steps, "predicted": "1" if answered else None,
        "answered": answered, "grounded": grounded, "correct": correct, "efficiency": efficiency, "reward": reward,
        "cost": {"tool_calls": tool_calls, "sql_calls": tool_calls, "failed_calls": failed, "rows_returned": rows,
                 "vm_steps": vm},
        "optimal": {"tool_calls": optimal[0], "rows": optimal[1], "vm_steps": optimal[2]},
        "raw_completion": None, "completion_tokens": tokens, "latency_s": latency,
    }


def three_episodes():
    good = make_trace("a", level=1, efficiency=1.0, reward=1.0)
    wasteful = make_trace("b", level=2, tool_calls=4, failed=1, rows=100, vm=300, efficiency=0.5, reward=0.75,
                          latency=2.0, tokens=50)
    wrong = make_trace("c", level=3, correct=False, answered=False, grounded=False, tool_calls=0, rows=0, vm=0,
                       efficiency=0.0, reward=0.0, final=False, latency=4.0, tokens=70)
    return [good, wasteful, wrong]


def test_aggregate_hand_computed():
    agg = M.aggregate(three_episodes())
    m, c = agg["metrics"], agg["counts"]
    assert agg["n"] == 3
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["answered_rate"] == pytest.approx(2 / 3)
    assert m["grounded_rate"] == pytest.approx(2 / 3)
    assert m["reward"] == pytest.approx((1.0 + 0.75 + 0.0) / 3)
    assert m["efficiency"] == pytest.approx(0.5)
    # only the two correct episodes: (1.0 + 0.5) / 2
    assert m["efficiency_when_correct"] == pytest.approx(0.75)
    assert c["efficiency_when_correct"] == 2
    assert m["tool_calls"] == pytest.approx(2.0)  # (2 + 4 + 0) / 3
    assert m["rows_returned"] == pytest.approx(101 / 3)
    assert m["vm_steps"] == pytest.approx(400 / 3)
    # call ratio: 2/2, 4/2, 0/2
    assert m["call_ratio"] == pytest.approx((1.0 + 2.0 + 0.0) / 3)
    # vm ratio uses +1: 101/101, 301/101, 1/101
    assert m["vm_step_ratio"] == pytest.approx((1 + 301 / 101 + 1 / 101) / 3)
    # tool-call validity is over all calls: 1 failed of 6 calls
    assert m["tool_call_validity"] == pytest.approx(1 - 1 / 6)
    assert m["no_failed_call_rate"] == pytest.approx(2 / 3)  # the episode with no calls has zero failed calls
    # correct and optimal: only "a" (calls 2 <= 2). "b" is correct but used 4 calls.
    assert m["correct_and_optimal_rate"] == pytest.approx(1 / 3)
    # latency and tokens exist for two of three episodes
    assert m["latency_s"] == pytest.approx(3.0)
    assert c["latency_s"] == 2
    assert m["completion_tokens"] == pytest.approx(60.0)


def test_null_latency_gives_none_not_zero():
    agg = M.aggregate([make_trace("a"), make_trace("b")])
    assert agg["metrics"]["latency_s"] is None
    assert agg["counts"]["latency_s"] == 0
    assert agg["metrics"]["completion_tokens"] is None


def test_no_correct_episode_gives_none_for_conditional_efficiency():
    agg = M.aggregate([make_trace("a", correct=False, efficiency=0.3, reward=0.1)])
    assert agg["metrics"]["efficiency_when_correct"] is None
    assert agg["metrics"]["efficiency"] == pytest.approx(0.3)  # the unconditional mean still counts it


def test_no_tool_calls_at_all_has_no_validity():
    agg = M.aggregate([make_trace("a", correct=False, answered=False, grounded=False, tool_calls=0, final=False)])
    assert agg["metrics"]["tool_call_validity"] is None


def test_every_metric_has_a_direction_and_label():
    for key in M.ALL_METRIC_KEYS:
        assert M.metric_direction(key) in ("max", "min")
        assert M.metric_label(key)
    assert M.metric_direction("accuracy") == "max"
    assert M.metric_direction("tool_calls") == "min"
    assert M.metric_direction("vm_step_ratio") == "min"


def test_breakdown_by_level_and_domain():
    traces = three_episodes()
    by_level = M.breakdown(traces, "level")
    assert set(by_level) == {"1", "2", "3"}
    assert by_level["3"]["accuracy"] == 0.0 and by_level["3"]["n"] == 1
    traces[1]["domain"] = "clinic"
    by_domain = M.breakdown(traces, "domain")
    assert by_domain["retail"]["n"] == 2
    assert by_domain["retail"]["tool_calls"] == pytest.approx(1.0)


# --- validation -----------------------------------------------------------

def test_valid_trace_has_no_problems():
    assert M.validate_trace(make_trace()) == []


@pytest.mark.parametrize("change, expected", [
    (lambda t: t.pop("cost"), "cost"),
    (lambda t: t.update(level=4), "level"),
    (lambda t: t.update(split="dev"), "split"),
    (lambda t: t.update(correct=1), "correct"),  # an int is not a bool
    (lambda t: t.update(efficiency=1.5), "efficiency"),
    (lambda t: t.update(latency_s="fast"), "latency_s"),
    (lambda t: t["cost"].update(tool_calls=5), "tool_call steps"),
    (lambda t: t["cost"].update(failed_calls=3), "cannot exceed"),
    (lambda t: t["optimal"].update(tool_calls=0), "optimal.tool_calls"),
    (lambda t: t["steps"].append({"type": "thought"}), "type"),
    (lambda t: t.update(answered=False), "correct is true"),
])
def test_validation_catches_problems(change, expected):
    trace = copy.deepcopy(make_trace())
    change(trace)
    problems = M.validate_trace(trace)
    assert problems, "expected at least one problem"
    assert any(expected in p for p in problems), problems


def test_validation_rejects_non_object():
    assert M.validate_trace([1, 2]) == ["record is not a JSON object"]


# --- paired bootstrap -----------------------------------------------------

def test_bootstrap_identical_values_gives_zero_interval():
    r = M.paired_bootstrap_ci([1.0] * 10, [1.0] * 10, n_boot=200, seed=1)
    assert r["delta"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["excludes_zero"] is False


def test_bootstrap_clear_gain_excludes_zero():
    base = [0.0, 1.0] * 20
    cand = [x + 0.5 for x in base]
    r = M.paired_bootstrap_ci(base, cand, n_boot=500, seed=1)
    assert r["delta"] == pytest.approx(0.5)
    assert r["excludes_zero"] is True
    assert r["ci_low"] == pytest.approx(0.5) and r["ci_high"] == pytest.approx(0.5)  # every difference is 0.5


def test_bootstrap_noisy_difference_can_include_zero():
    base = [0.0] * 10
    cand = [1.0, -1.0] * 5
    r = M.paired_bootstrap_ci(base, cand, n_boot=1000, seed=3)
    assert r["delta"] == pytest.approx(0.0)
    assert r["ci_low"] < 0 < r["ci_high"]
    assert r["excludes_zero"] is False


def test_bootstrap_is_deterministic_for_a_seed():
    base, cand = [0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0]
    assert M.paired_bootstrap_ci(base, cand, 300, seed=7) == M.paired_bootstrap_ci(base, cand, 300, seed=7)


def test_bootstrap_input_errors():
    with pytest.raises(ValueError):
        M.paired_bootstrap_ci([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        M.paired_bootstrap_ci([], [])
    with pytest.raises(ValueError):
        M.paired_bootstrap_ci([1.0], [1.0], n_boot=0)


def test_paired_from_traces_pairs_by_task_id_and_uses_shared_ids_only():
    base = [make_trace("a", reward=0.5), make_trace("b", reward=0.5), make_trace("only-base", reward=0.0)]
    cand = [make_trace("b", reward=1.0), make_trace("a", reward=1.0), make_trace("only-cand", reward=0.0)]  # different order
    r = M.paired_from_traces(base, cand, "reward", n_boot=100, seed=0)
    assert r["n"] == 2
    assert r["delta"] == pytest.approx(0.5)
