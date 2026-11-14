"""GRPO script pieces that run without a model: new GRPOConfig flags, the no-signal guard."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

pytest.importorskip("trl")

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def grpo():
    spec = importlib.util.spec_from_file_location("train_grpo", ROOT / "scripts" / "train_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Flags reach GRPOConfig
# ---------------------------------------------------------------------------


def test_new_flags_are_grpo_config_fields():
    # AgentTune's _split_kwargs keeps a kwarg only if it is in this signature.
    from trl import GRPOConfig

    fields = set(inspect.signature(GRPOConfig.__init__).parameters)
    assert {"top_p", "scale_rewards", "loss_type", "epsilon_high"} <= fields


@pytest.mark.parametrize("scale", ["group", "batch", "none"])
def test_scale_rewards_values_are_accepted(scale, tmp_path):
    from trl import GRPOConfig

    cfg = GRPOConfig(output_dir=str(tmp_path), scale_rewards=scale, use_cpu=True, report_to="none")
    assert cfg.scale_rewards == scale


def test_other_values_reach_the_config(tmp_path):
    from trl import GRPOConfig

    cfg = GRPOConfig(output_dir=str(tmp_path), top_p=0.9, loss_type="dr_grpo", epsilon_high=0.28,
                     use_cpu=True, report_to="none")
    assert (cfg.top_p, cfg.loss_type, cfg.epsilon_high) == (0.9, "dr_grpo", 0.28)


def test_defaults_keep_the_old_behaviour(grpo):
    from trl import GRPOConfig

    parser_defaults = vars(grpo.build_parser().parse_args([]))
    trl_defaults = {f.name: f.default for f in GRPOConfig.__dataclass_fields__.values()}
    assert parser_defaults["scale_rewards"] == trl_defaults["scale_rewards"] == "group"
    assert parser_defaults["loss_type"] == trl_defaults["loss_type"] == "dapo"
    assert parser_defaults["epsilon_high"] is trl_defaults["epsilon_high"] is None
    assert parser_defaults["top_p"] == 0.95  # AgentTune's default (agentic_grpo.py:423), used by the old runs
    assert parser_defaults["stop_if_no_signal"] == 3
    assert parser_defaults["max_tool_steps"] == 4


# ---------------------------------------------------------------------------
# Group spread metric
# ---------------------------------------------------------------------------


def test_group_spread_metric(grpo):
    seen = {}
    base = lambda completions, **kw: [1.0, 1.0, 0.0, 1.0, 0.1, 0.1]  # noqa: E731
    fn = grpo.with_group_spread_metric(base, num_generations=2)
    out = fn(["a"] * 6, log_metric=lambda k, v: seen.__setitem__(k, v))
    assert out == [1.0, 1.0, 0.0, 1.0, 0.1, 0.1]
    assert seen == {"frugal/frac_groups_no_spread": pytest.approx(2 / 3)}


def test_group_spread_metric_without_logger(grpo):
    fn = grpo.with_group_spread_metric(lambda completions, **kw: [0.5, 0.5], num_generations=2)
    assert fn(["a", "b"]) == [0.5, 0.5]


# ---------------------------------------------------------------------------
# No-signal guard
# ---------------------------------------------------------------------------


def step(value, key="frugal/frac_groups_no_spread"):
    return {"loss": 0.0, "reward": 1.0, "frac_reward_zero_std": 1.0, key: value}


def test_guard_stops_after_n_flat_steps(grpo):
    guard = grpo.NoSignalGuard(3)
    assert not guard.update(step(1.0))
    assert not guard.update(step(1.0))
    assert guard.update(step(1.0))
    assert guard.stopped and guard.values == [1.0, 1.0, 1.0]


def test_guard_keeps_going_with_any_signal(grpo):
    guard = grpo.NoSignalGuard(3)
    for value in (1.0, 0.75, 1.0):
        assert not guard.update(step(value))
    # Only the first N steps count: later flat steps never stop the run.
    for _ in range(5):
        assert not guard.update(step(1.0))
    assert not guard.stopped


def test_guard_off_and_non_step_lines(grpo):
    off = grpo.NoSignalGuard(0)
    assert not any(off.update(step(1.0)) for _ in range(5))
    guard = grpo.NoSignalGuard(1)
    assert not guard.update({"train_runtime": 12.0, "train_loss": 0.0})  # final summary line
    assert guard.values == []


def test_guard_falls_back_to_trl_metric(grpo):
    guard = grpo.NoSignalGuard(1)
    assert guard.update({"reward": 1.0, "frac_reward_zero_std": 1.0})


def test_callback_writes_log_and_stops_training(grpo, tmp_path):
    from transformers import TrainerControl, TrainerState

    log = tmp_path / "reward_log.jsonl"
    guard = grpo.NoSignalGuard(2)
    cb = grpo.make_reward_logger(log, guard)
    state, control = TrainerState(), TrainerControl()
    for n in (1, 2):
        state.global_step = n
        cb.on_log(None, state, control, logs=step(1.0))
    assert control.should_training_stop
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x["step"] for x in lines] == [1, 2]
    assert all("frugal/frac_groups_no_spread" in x and "frac_reward_zero_std" in x for x in lines)


def test_callback_does_not_stop_with_signal(grpo, tmp_path):
    from transformers import TrainerControl, TrainerState

    cb = grpo.make_reward_logger(tmp_path / "r.jsonl", grpo.NoSignalGuard(2))
    state, control = TrainerState(), TrainerControl()
    cb.on_log(None, state, control, logs=step(1.0))
    cb.on_log(None, state, control, logs=step(0.5))
    assert not control.should_training_stop


def test_exit_code_is_distinct(grpo):
    assert grpo.EXIT_NO_SIGNAL not in (0, 1, 2)
