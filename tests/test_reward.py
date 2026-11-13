import pytest

from frugal_analyst.db import ensure_dbs
from frugal_analyst.policies import guess_policy, naive_policy, oracle_policy
from frugal_analyst.reward import (
    RewardConfig, ToolCall, answers_match, extract_answer, score_trajectory,
)
from frugal_analyst.tasks import generate_tasks

ensure_dbs()
GATED = RewardConfig()
UNGATED = RewardConfig(mode="ungated")


@pytest.fixture(scope="module")
def splits():
    return generate_tasks(0)


@pytest.fixture(scope="module")
def all_tasks(splits):
    return [t for tasks in splits.values() for t in tasks]


def first(splits, level, answer_type="number"):
    return next(t for t in splits["test_id"] if t.level == level and t.answer_type == answer_type)


# --- answer handling --------------------------------------------------------


def test_extract_answer():
    assert extract_answer("thinking...\nANSWER: 42") == "42"
    assert extract_answer("ANSWER: 42\n\n") == "42"
    assert extract_answer("ANSWER:   Garden  ") == "Garden"
    assert extract_answer("ANSWER: 42\nand some more text") is None  # must be the last line
    assert extract_answer("The answer is 42") is None
    assert extract_answer("ANSWER:") is None
    assert extract_answer("") is None
    assert extract_answer(None) is None


@pytest.mark.parametrize("predicted,gold,ok", [
    ("42", "42", True),
    ("42.0", "42", True),
    ("1,234.50", "1234.5", True),
    ("$1,234.50", "1234.5", True),
    ("12.5%", "12.5", True),
    ("1234.499", "1234.5", True),  # within 0.01
    ("1000000", "1000500", True),  # relative error 0.0005
    ("1000000", "1002000", False),
    ("43", "42", False),
    ("forty two", "42", False),
    ("", "42", False),
])
def test_number_matching(predicted, gold, ok):
    assert answers_match(predicted, gold, "number") is ok


@pytest.mark.parametrize("predicted,gold,ok", [
    ("Garden", "Garden", True),
    ("garden", "Garden", True),
    ("  Garden. ", "Garden", True),
    ('"Garden"', "Garden", True),
    ("'Garden'.", "Garden", True),
    ("Gardens", "Garden", False),
    ("Toys", "Garden", False),
])
def test_text_matching(predicted, gold, ok):
    assert answers_match(predicted, gold, "text") is ok


# --- required spec tests ----------------------------------------------------


def test_oracle_scores_one(splits):
    for level in (1, 2, 3):
        task = first(splits, level)
        score = score_trajectory(task, *oracle_policy(task))
        assert score.reward == 1.0 and score.correct and score.efficiency == 1.0
        assert score.grounded and score.answered and score.predicted == task.gold_answer
        assert score.cost.tool_calls == task.optimal_tool_calls
        assert score.cost.failed_calls == 0


def test_empty_trajectory_scores_zero(splits):
    task = first(splits, 1)
    score = score_trajectory(task, [], None)
    assert score.reward == 0.0
    assert (score.answered, score.grounded, score.correct, score.efficiency) == (False, False, False, 0.0)
    assert score.cost.tool_calls == 0


def test_select_star_dump_scores_below_oracle(splits):
    task = first(splits, 1)
    calls, text = oracle_policy(task)
    dump = [calls[0], ToolCall("run_sql", {"database": task.domain, "query": "SELECT * FROM orders" if task.domain == "retail" else f"SELECT * FROM {task.gold_sql.split(' FROM ')[1].split()[0]}"}), calls[1]]
    score = score_trajectory(task, dump, text)
    assert score.correct
    assert 0.5 < score.reward < 1.0
    assert score.efficiency < 1.0
    assert score.cost.rows_returned > task.optimal_rows


def test_correct_number_without_a_query_is_not_rewarded(splits):
    task = first(splits, 1)
    score = score_trajectory(task, [], f"ANSWER: {task.gold_answer}")
    assert score.answered and not score.grounded and not score.correct
    assert score.reward <= 0.05
    # A schema call is a valid call but not grounding: still at most 0.05.
    schema_only = [ToolCall("get_schema", {"database": task.domain})]
    score = score_trajectory(task, schema_only, f"ANSWER: {task.gold_answer}")
    assert not score.correct and score.reward == 0.05


# --- reward shape -----------------------------------------------------------


def test_grounded_wrong_answer_gets_a_small_reward(splits):
    task = first(splits, 1)
    calls, _ = oracle_policy(task)
    score = score_trajectory(task, calls, "ANSWER: -1")
    assert score.grounded and score.answered and not score.correct and score.reward == 0.1


def test_grounded_but_no_answer_line(splits):
    task = first(splits, 1)
    calls, _ = oracle_policy(task)
    score = score_trajectory(task, calls, "I think it is big")
    assert score.grounded and not score.answered and score.reward == 0.05


def test_failed_calls_are_counted_and_cost_efficiency(splits):
    task = first(splits, 1)
    calls, text = oracle_policy(task)
    bad = ToolCall("run_sql", {"database": task.domain, "query": "SELEC oops"})
    clean = score_trajectory(task, calls, text)
    messy = score_trajectory(task, [calls[0], bad, calls[1]], text)
    assert messy.cost.failed_calls == 1 and messy.cost.sql_calls == 2 and messy.correct
    assert messy.efficiency < clean.efficiency


def test_a_write_attempt_is_a_failed_call(splits):
    task = first(splits, 1)
    calls, text = oracle_policy(task)
    attack = ToolCall("run_sql", {"database": task.domain, "query": "DELETE FROM " + task.gold_sql.split(" FROM ")[1].split()[0]})
    score = score_trajectory(task, [attack] + calls, text)
    assert score.cost.failed_calls == 1 and score.correct


def test_reward_ignores_tool_output_text_and_is_deterministic(splits):
    task = first(splits, 2)
    a = score_trajectory(task, *oracle_policy(task))
    b = score_trajectory(task, *oracle_policy(task))
    assert a == b


def test_unknown_tool_and_bad_arguments_do_not_raise(splits):
    task = first(splits, 1)
    calls = [ToolCall("drop_database", {}), ToolCall("run_sql", {}), ToolCall("run_sql", {"database": task.domain, "query": 5})]
    score = score_trajectory(task, calls, "ANSWER: 1")
    assert score.cost.failed_calls == 3 and score.reward == 0.0


# --- scripted policies on every task ---------------------------------------


def test_oracle_is_perfect_on_every_task(all_tasks):
    assert len(all_tasks) == 660
    for task in all_tasks:
        score = score_trajectory(task, *oracle_policy(task))
        assert score.reward == 1.0 and score.correct, task.id


def test_naive_is_correct_but_inefficient_on_every_task(all_tasks):
    for task in all_tasks:
        score = score_trajectory(task, *naive_policy(task))
        assert score.correct, task.id
        assert 0.0 < score.efficiency < 1.0, task.id
        assert 0.5 < score.reward < 1.0, task.id


def test_guess_is_worthless_when_gated(all_tasks):
    for task in all_tasks:
        score = score_trajectory(task, *guess_policy(task), cfg=GATED)
        assert score.reward <= 0.05 and not score.correct, task.id


# --- the reward hacking demonstration --------------------------------------


def test_ungated_reward_pays_for_not_querying(splits):
    """Under the ungated reward, skipping the query beats an honest wrong answer."""
    task = first(splits, 1)
    guess_calls, guess_text = guess_policy(task)
    honest_calls, _ = naive_policy(task)  # a real attempt that grounds itself, then gets the answer wrong
    wrong = "ANSWER: -1"

    hack = score_trajectory(task, guess_calls, guess_text, UNGATED)
    honest = score_trajectory(task, honest_calls, wrong, UNGATED)
    assert honest.grounded and not honest.correct and not hack.grounded
    assert hack.reward > honest.reward

    # The gated reward reverses the order.
    hack_g = score_trajectory(task, guess_calls, guess_text, GATED)
    honest_g = score_trajectory(task, honest_calls, wrong, GATED)
    assert honest_g.reward > hack_g.reward


def test_ungated_still_prefers_correct_answers(all_tasks):
    task = all_tasks[0]
    ok = score_trajectory(task, *oracle_policy(task), cfg=UNGATED)
    hack = score_trajectory(task, *guess_policy(task), cfg=UNGATED)
    assert ok.reward == 1.0 and hack.reward == 0.5


def test_ungated_no_query_beats_honest_wrong_on_every_task(all_tasks):
    for task in all_tasks:
        hack = score_trajectory(task, *guess_policy(task), cfg=UNGATED)
        honest = score_trajectory(task, naive_policy(task)[0], "ANSWER: -1", UNGATED)
        assert hack.reward > honest.reward, task.id
