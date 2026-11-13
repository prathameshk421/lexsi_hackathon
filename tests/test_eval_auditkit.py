"""Tests for scripts/eval_auditkit.py. They build small synthetic trace files in a tmp dir and run the real script."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("eval_auditkit", ROOT / "scripts" / "eval_auditkit.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

import auditkit as ak  # noqa: E402


def make_trace(task_id, model, split="test_id", level=1, domain="retail", correct=True, tool_calls=2, failed=0,
               rows=1, vm=100, optimal=(2, 1, 100), efficiency=1.0, reward=1.0, latency=None, tokens=None):
    steps = [{"type": "tool_call", "name": "run_sql", "arguments": {"database": domain, "query": "SELECT 1"},
              "ok": i >= failed, "n_rows": 1, "vm_steps": 10, "output": "x"} for i in range(tool_calls)]
    steps.append({"type": "final", "text": "ANSWER: 1"})
    return {
        "task_id": task_id, "domain": domain, "level": level, "split": split, "question": f"question {task_id}?",
        "gold_answer": "1", "answer_type": "number", "model": model, "steps": steps,
        "predicted": "1", "answered": True, "grounded": True, "correct": correct,
        "efficiency": efficiency, "reward": reward,
        "cost": {"tool_calls": tool_calls, "sql_calls": tool_calls, "failed_calls": failed, "rows_returned": rows,
                 "vm_steps": vm},
        "optimal": {"tool_calls": optimal[0], "rows": optimal[1], "vm_steps": optimal[2]},
        "raw_completion": None, "completion_tokens": tokens, "latency_s": latency,
    }


def write(path, traces):
    path.write_text("".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8")
    return path


def good_model(ids):
    return [make_trace(i, "grpo", level=1 + n % 3, efficiency=1.0, reward=1.0) for n, i in enumerate(ids)]


def wasteful_model(ids, label="zero-shot", latency=None):
    # Correct for the first 3 ids, wrong for the rest. Always pays 4 calls (1 failed) and many rows.
    out = []
    for n, i in enumerate(ids):
        ok = n < 3
        out.append(make_trace(i, label, level=1 + n % 3, correct=ok, tool_calls=4, failed=1, rows=200, vm=500,
                              efficiency=0.4 if ok else 0.2, reward=0.7 if ok else 0.1, latency=latency,
                              tokens=None if latency is None else 100))
    return out


IDS = [f"retail-L1-{n:04d}" for n in range(6)]


def run_script(tmp_path, files, *extra):
    out = tmp_path / "out"
    code = script.main(["--traces", *[str(f) for f in files], "--out", str(out), "--bootstrap", "300", *extra])
    return code, out


def load(out):
    return json.loads((out / "results.json").read_text())


def test_end_to_end_numbers_and_comparison(tmp_path):
    f1 = write(tmp_path / "zs.jsonl", wasteful_model(IDS))
    f2 = write(tmp_path / "grpo.jsonl", good_model(IDS))
    code, out = run_script(tmp_path, [f1, f2])
    assert code == 0
    res = load(out)
    assert res["baseline"] == "zero-shot"  # default when present
    assert res["auditkit_version"] == ak.__version__
    assert res["auditkit_failed_samples_total"] == 0 and res["crosscheck_ok"] is True
    assert {i["path"] for i in res["inputs"]} == {str(f1), str(f2)}
    assert all(len(i["sha256"]) == 64 for i in res["inputs"])

    zs = res["splits"]["test_id"]["models"]["zero-shot"]
    m = zs["metrics"]
    assert zs["n"] == 6
    assert m["accuracy"] == pytest.approx(3 / 6)
    assert m["reward"] == pytest.approx((3 * 0.7 + 3 * 0.1) / 6)
    assert m["efficiency_when_correct"] == pytest.approx(0.4)  # correct episodes only
    assert zs["counts"]["efficiency_when_correct"] == 3
    assert m["tool_calls"] == pytest.approx(4.0)
    assert m["tool_call_validity"] == pytest.approx(0.75)  # 1 failed of 4 calls
    assert m["call_ratio"] == pytest.approx(2.0)
    assert m["correct_and_optimal_rate"] == 0.0
    assert zs["by_level"]["1"]["n"] == 2

    g = res["splits"]["test_id"]["models"]["grpo"]["metrics"]
    assert g["accuracy"] == 1.0 and g["efficiency"] == 1.0 and g["call_ratio"] == 1.0
    assert g["correct_and_optimal_rate"] == 1.0

    pair = res["comparisons"]["test_id"]["pairs"][0]
    assert pair["model"] == "grpo" and pair["n_paired"] == 6
    acc = pair["metrics"]["accuracy"]
    assert acc["delta"] == pytest.approx(0.5)
    assert acc["ci_low"] <= acc["delta"] <= acc["ci_high"]
    assert pair["metrics"]["tool_calls"]["delta"] == pytest.approx(-2.0)
    assert pair["metrics"]["tool_calls"]["verdict"] == "better"  # fewer calls is better
    assert acc["auditkit_p_value"] is not None
    assert acc["auditkit_delta"] == pytest.approx(0.5)

    md = (out / "results.md").read_text()
    assert "Split `test_id`" in md and "`grpo`" in md and "Paired comparisons" in md
    assert chr(0x2014) not in md  # no em dashes


def test_null_latency_is_skipped_and_said_so(tmp_path):
    f1 = write(tmp_path / "a.jsonl", wasteful_model(IDS))  # latency null everywhere
    f2 = write(tmp_path / "b.jsonl", wasteful_model(IDS, label="grpo", latency=1.5))
    code, out = run_script(tmp_path, [f1, f2])
    assert code == 0
    res = load(out)
    zs = res["splits"]["test_id"]["models"]["zero-shot"]
    assert zs["metrics"]["latency_s"] is None and zs["counts"]["latency_s"] == 0
    assert zs["auditkit_failed_samples"] == 0  # a left-out value is not a failure
    other = res["splits"]["test_id"]["models"]["grpo"]
    assert other["metrics"]["latency_s"] == pytest.approx(1.5)
    assert other["metrics"]["completion_tokens"] == pytest.approx(100)
    md = (out / "results.md").read_text()
    assert "n/a" in md


def test_malformed_record_fails_loudly_with_file_and_line(tmp_path, capsys):
    bad = make_trace("x-2", "zero-shot")
    del bad["cost"]
    f = tmp_path / "bad.jsonl"
    write(f, [make_trace("x-1", "zero-shot"), bad])
    code, out = run_script(tmp_path, [f])
    assert code == script.EXIT_BAD_INPUT
    err = capsys.readouterr().err
    assert "bad.jsonl:2" in err and "cost" in err
    assert not (out / "results.json").exists()


def test_invalid_json_line_fails_with_line_number(tmp_path, capsys):
    f = tmp_path / "broken.jsonl"
    f.write_text(json.dumps(make_trace("a", "zero-shot")) + "\n{not json\n", encoding="utf-8")
    code, _ = run_script(tmp_path, [f])
    assert code == script.EXIT_BAD_INPUT
    assert "broken.jsonl:2" in capsys.readouterr().err


def test_duplicate_task_id_is_an_error(tmp_path, capsys):
    f = write(tmp_path / "dup.jsonl", [make_trace("a", "zero-shot"), make_trace("a", "zero-shot")])
    code, _ = run_script(tmp_path, [f])
    assert code == script.EXIT_BAD_INPUT
    assert "duplicate task_id" in capsys.readouterr().err


def test_missing_ids_are_reported_and_pairs_use_shared_ids(tmp_path):
    f1 = write(tmp_path / "zs.jsonl", wasteful_model(IDS))
    f2 = write(tmp_path / "grpo.jsonl", good_model(IDS[:4]))  # grpo lacks the last two ids
    code, out = run_script(tmp_path, [f1, f2])
    assert code == 0
    res = load(out)
    cov = res["splits"]["test_id"]["coverage"]
    assert cov["union"] == 6 and cov["shared"] == 4 and cov["complete"] is False
    assert cov["missing"] == {"grpo": IDS[4:]}
    pair = res["comparisons"]["test_id"]["pairs"][0]
    assert pair["n_paired"] == 4
    # On the 4 shared ids zero-shot is right for 3 of 4 and grpo for 4 of 4.
    assert pair["metrics"]["accuracy"]["baseline_mean"] == pytest.approx(3 / 4)
    assert pair["metrics"]["accuracy"]["delta"] == pytest.approx(0.25)
    # The per-model table still uses each model's own episodes.
    assert res["splits"]["test_id"]["models"]["zero-shot"]["n"] == 6
    assert "missing 2 task ids" in (out / "results.md").read_text()


def test_baseline_defaults_to_naive_without_zero_shot(tmp_path):
    f1 = write(tmp_path / "naive.jsonl", wasteful_model(IDS, label="naive"))
    f2 = write(tmp_path / "oracle.jsonl", [make_trace(i, "oracle") for i in IDS])
    code, out = run_script(tmp_path, [f1, f2])
    assert code == 0
    res = load(out)
    assert res["baseline"] == "naive"
    assert [p["model"] for p in res["comparisons"]["test_id"]["pairs"]] == ["oracle"]


def test_unknown_baseline_is_an_error(tmp_path, capsys):
    f = write(tmp_path / "a.jsonl", wasteful_model(IDS))
    code, _ = run_script(tmp_path, [f], "--baseline", "nope")
    assert code == script.EXIT_BAD_INPUT
    assert "nope" in capsys.readouterr().err


def test_baseline_missing_on_one_split_is_noted(tmp_path):
    f1 = write(tmp_path / "zs.jsonl", wasteful_model(IDS))  # test_id only
    ood = [make_trace(f"saas-{n}", "grpo", split="test_ood", domain="saas") for n in range(3)]
    f2 = write(tmp_path / "ood.jsonl", ood)
    code, out = run_script(tmp_path, [f1, f2])
    assert code == 0
    comp = load(out)["comparisons"]["test_ood"]
    assert comp["pairs"] == [] and "no traces on this split" in comp["note"]


def test_failed_samples_make_exit_code_nonzero(tmp_path, monkeypatch):
    def boom(sample, output):
        if sample.id == IDS[0]:
            raise ValueError("scorer bug")
        return 1.0

    extra = ak.scorer(boom, direction=ak.Direction.MAXIMIZE, name="boom")
    monkeypatch.setattr(script, "SCORERS", script.SCORERS + [extra])
    f = write(tmp_path / "a.jsonl", wasteful_model(IDS))
    code, out = run_script(tmp_path, [f])
    assert code == script.EXIT_FAILED_SAMPLES
    res = load(out)
    assert res["auditkit_failed_samples_total"] == 1  # still written, so the failure is visible
    assert res["splits"]["test_id"]["models"]["zero-shot"]["auditkit_errors"][0]["metric"] == "boom"


def test_crosscheck_catches_a_scorer_that_disagrees(tmp_path, monkeypatch):
    # A scorer that quietly changes accuracy must not pass the cross-check against the stored `correct` field.
    lying = ak.scorer(lambda s, o: 1.0, direction=ak.Direction.MAXIMIZE, name="accuracy")
    monkeypatch.setattr(script, "SCORERS", [lying] + script.SCORERS[1:])
    f = write(tmp_path / "a.jsonl", wasteful_model(IDS))
    code, out = run_script(tmp_path, [f])
    assert code == script.EXIT_MISMATCH
    assert load(out)["crosscheck_ok"] is False


def test_glob_pattern_is_expanded(tmp_path):
    write(tmp_path / "x_1.jsonl", wasteful_model(IDS))
    out = tmp_path / "out"
    code = script.main(["--traces", str(tmp_path / "x_*.jsonl"), "--out", str(out), "--bootstrap", "50"])
    assert code == 0


def test_runs_on_the_real_scripted_traces_if_present():
    traces = sorted((ROOT / "outputs" / "traces").glob("*_test_*.jsonl")) if (ROOT / "outputs" / "traces").exists() else []
    if not any(p.name.startswith("oracle") for p in traces):
        pytest.skip("scripted trace files not generated")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        code = script.main(["--traces", *map(str, traces), "--out", d, "--bootstrap", "100"])
        assert code == 0
        res = json.loads((Path(d) / "results.json").read_text())
    for split in res["splits"].values():
        oracle = split["models"]["oracle"]["metrics"]
        assert oracle["accuracy"] == 1.0 and oracle["efficiency"] == 1.0 and oracle["call_ratio"] == 1.0
