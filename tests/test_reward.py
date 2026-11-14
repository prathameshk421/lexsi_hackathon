import pytest

from frugal_analyst.db import ensure_dbs
from frugal_analyst.env import run_tool
from frugal_analyst.policies import _tables_in, guess_policy, naive_policy, oracle_policy
from frugal_analyst.reward import (
    RewardConfig, ToolCall, answers_match, extract_answer, find_support, score_trajectory, select_list_literals,
    sql_literals,
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


# --- evidence rule: the answer must come from a query that read data --------


@pytest.fixture(scope="module")
def train_val(splits):
    return splits["train"] + splits["val"]


def table_of(task):
    return _tables_in(task.gold_sql)[0]


def lit(value):
    return "'" + value.replace("'", "''") + "'"


def wrong(task):
    if task.answer_type == "number":
        gold = float(task.gold_answer)
        return str(gold + max(7.0, abs(gold) * 0.5))  # far outside the match tolerance
    return "no such value"


def run(task, *queries):
    return [ToolCall("get_schema", {"database": task.domain})] + [
        ToolCall("run_sql", {"database": task.domain, "query": q}) for q in queries]


def test_oracle_is_perfect_and_supported_on_every_train_and_val_task(train_val):
    assert len(train_val) == 440
    for task in train_val:
        score = score_trajectory(task, *oracle_policy(task))
        assert score.reward == 1.0 and score.correct and score.supported and score.grounded, task.id


def test_select_1_then_gold_is_not_rewarded(train_val):
    for task in train_val:
        score = score_trajectory(task, run(task, "SELECT 1"), f"ANSWER: {task.gold_answer}")
        assert not score.grounded and not score.correct and score.reward <= 0.05, task.id


def test_gold_typed_as_a_literal_is_not_rewarded(train_val):
    for task in train_val:
        score = score_trajectory(task, run(task, f"SELECT {lit(task.gold_answer)}"), f"ANSWER: {task.gold_answer}")
        assert not score.grounded and not score.correct and score.reward <= 0.05, task.id


def test_gold_literal_from_a_real_table_is_grounded_but_not_correct(train_val):
    for task in train_val:
        query = f"SELECT {lit(task.gold_answer)} FROM {table_of(task)} LIMIT 1"
        score = score_trajectory(task, run(task, query), f"ANSWER: {task.gold_answer}")
        assert score.grounded and not score.supported and not score.correct, task.id
        assert score.reward <= 0.1, task.id


def test_wrong_answer_after_select_1_scores_005(train_val):
    for task in train_val:
        score = score_trajectory(task, run(task, "SELECT 1"), f"ANSWER: {wrong(task)}")
        assert score.reward == 0.05, task.id


def test_honest_wrong_attempt_scores_01(train_val):
    for task in train_val:
        score = score_trajectory(task, oracle_policy(task)[0], f"ANSWER: {wrong(task)}")
        assert score.grounded and not score.correct and score.reward == 0.1, task.id


def test_oracle_plus_preview_is_between_naive_and_oracle(train_val):
    for task in train_val:
        calls, text = oracle_policy(task)
        preview = [calls[0], ToolCall("run_sql", {"database": task.domain,
                                                  "query": f"SELECT * FROM {table_of(task)} LIMIT 5"})] + calls[1:]
        p = score_trajectory(task, preview, text).reward
        naive = score_trajectory(task, *naive_policy(task)).reward
        assert naive < p < 1.0, task.id


def test_answer_literal_in_where_still_scores_one(splits):
    # 2 of 400 train tasks filter on a number that equals the answer. That is real work, not a cheat.
    tasks = {t.id: t for t in splits["train"]}
    for task_id in ("school-L3-0012", "school-L3-0039"):
        task = tasks[task_id]
        assert task.gold_answer in sql_literals(task.gold_sql)
        assert task.gold_answer not in select_list_literals(task.gold_sql)
        score = score_trajectory(task, *oracle_policy(task))
        assert score.reward == 1.0 and score.supported
        assert score.answer_in_sql_literal  # the suspicion flag fires: it is a metric, not a penalty


def hidden_literal_cheats(task):
    """Queries that read a real table but take the answer from a literal in some select list."""
    t, g = table_of(task), lit(task.gold_answer)
    cheats = {
        "cte": f"WITH g(v) AS (SELECT {g}) SELECT v FROM g, {t} LIMIT 1",
        "cte with FROM": f"WITH g AS (SELECT {g} AS v FROM {t} LIMIT 1) SELECT v FROM g",
        "from subquery": f"SELECT v FROM (SELECT {g} AS v), {t} LIMIT 1",
        "values": f"WITH x(v) AS (VALUES ({g})) SELECT v FROM x, {t} LIMIT 1",
        "scalar subquery": f"SELECT (SELECT {g}) FROM {t} LIMIT 1",
        "case": f"SELECT CASE WHEN 1 THEN {g} END FROM {t} LIMIT 1",
        "function": f"SELECT MAX({g}) FROM {t}",
        "union": f"SELECT 1 FROM {t} WHERE 0 UNION SELECT {g}",
        "union values": f"SELECT 1 FROM {t} WHERE 0 UNION ALL VALUES ({g})",
        "in a WHERE subquery": f"SELECT v FROM (SELECT {g} AS v) WHERE v IN (SELECT {g} FROM {t})",
    }
    if task.answer_type == "number":  # an unquoted number is valid SQL only for number answers
        cheats["bare number"] = f"SELECT {task.gold_answer} AS n FROM {t} LIMIT 1"
    return cheats


def test_literal_in_any_select_list_is_not_supported(train_val):
    for task in train_val:
        for name, query in hidden_literal_cheats(task).items():
            score = score_trajectory(task, run(task, query), f"ANSWER: {task.gold_answer}")
            assert score.grounded and not score.supported and not score.correct, (task.id, name)
            assert score.reward <= 0.1, (task.id, name)


def test_oracle_is_supported_on_the_test_splits_too(splits):
    # A sanity check of the rule on data it was not built on. Not used for tuning.
    for task in splits["test_id"] + splits["test_ood"]:
        score = score_trajectory(task, *oracle_policy(task))
        assert score.reward == 1.0 and score.supported and not score.answer_in_sql_literal, task.id


def test_where_echo_is_supported_but_flagged(splits):
    """Known residual gap. A filter value echoed back through a column is support,
    because the 2 honest WHERE-literal tasks look the same. Only the flag shows it."""
    task = next(t for t in splits["train"] if t.id == "school-L3-0039")
    query = f"SELECT grade_level FROM students WHERE grade_level = {task.gold_answer} LIMIT 1"
    score = score_trajectory(task, run(task, query), f"ANSWER: {task.gold_answer}")
    assert score.correct and score.answer_in_sql_literal


def test_honest_query_next_to_a_cheat_still_counts(splits):
    task = first(splits, 1)
    calls, text = oracle_policy(task)
    cheat = ToolCall("run_sql", {"database": task.domain,
                                 "query": f"SELECT {lit(task.gold_answer)} FROM {table_of(task)} LIMIT 1"})
    score = score_trajectory(task, calls + [cheat], text)
    assert score.correct and score.supported and not score.answer_in_sql_literal


def test_a_comment_with_the_answer_does_not_matter(splits):
    task = first(splits, 1)
    calls, text = oracle_policy(task)
    query = calls[-1].arguments["query"].replace("SELECT", f"SELECT /* {task.gold_answer} */", 1)
    score = score_trajectory(task, calls[:-1] + [ToolCall("run_sql", {"database": task.domain, "query": query})], text)
    assert score.correct and not score.answer_in_sql_literal


def test_query_with_no_rows_does_not_support_a_lucky_answer(splits):
    # A real case from the distilled model: a wrong status code gave "(no rows)", and the answer was still gold.
    task = next(t for t in splits["train"] if t.level == 3 and t.answer_type == "text")
    empty = f"SELECT * FROM {table_of(task)} WHERE 0"
    score = score_trajectory(task, run(task, empty), f"ANSWER: {task.gold_answer}")
    assert score.grounded and not score.supported and not score.correct and score.reward == 0.1


def test_support_uses_every_fetched_row_not_only_the_20_shown(splits):
    task = next(t for t in splits["train"] if t.answer_type == "text")
    # 30 NULL rows first, then the real answer row: the answer is below the 20 shown rows.
    query = (f"SELECT * FROM (SELECT NULL FROM {table_of(task)} LIMIT 30) "
             f"UNION ALL SELECT * FROM ({task.gold_sql})")
    result = run_tool("run_sql", {"database": task.domain, "query": query})
    assert result.ok and result.n_rows > 20 and task.gold_answer not in result.output
    score = score_trajectory(task, run(task, query), f"ANSWER: {task.gold_answer}")
    assert score.supported and score.correct


def test_literal_helpers():
    q = "WITH t AS (SELECT 'x' AS a) SELECT COUNT(*), 'it''s', -5, 2.50, \"dq\" FROM t WHERE b = 'w' -- 'c'"
    assert select_list_literals(q) == ["x", "it's", "5", "-5", "2.50", "dq"]
    assert sql_literals(q) == ["x", "it's", "5", "-5", "2.50", "dq", "w"]
    assert select_list_literals("SELECT a FROM t UNION SELECT 7 FROM u ORDER BY 1") == ["7"]
    # The WHERE of a scalar subquery is not a select list. Neither is the outer WHERE.
    assert select_list_literals("SELECT (SELECT 3 FROM u WHERE z = 4) FROM t WHERE y = 9") == ["3"]
    # Filters, joins, IN lists, HAVING and LIMIT do not count. A subquery in WHERE has its own select list.
    q2 = ("SELECT COUNT(*) FROM a JOIN b ON b.k = a.k AND b.c = 'J' WHERE a.x IN (1, 2) "
          "AND a.y IN (SELECT 8 FROM c WHERE c.z = 'Z') GROUP BY a.g HAVING COUNT(*) > 3 ORDER BY 1 LIMIT 5")
    assert select_list_literals(q2) == ["8"]
    assert select_list_literals("WITH g(v) AS (VALUES ('a'), ('b')) SELECT v FROM g") == ["a", "b"]
    assert select_list_literals("SELECT CASE WHEN s = 'X' THEN 'yes' ELSE 'no' END FROM t") == ["X", "yes", "no"]
    assert select_list_literals("SELECT ROUND(AVG(v), 2) FROM t WHERE (a = 1 OR b = 2)") == ["2"]


def test_find_support_directly():
    assert find_support("42", "number", [("SELECT COUNT(*) FROM t", [(42,)])]) == (True, False)
    assert find_support("42", "number", [("SELECT 42 FROM t", [(42,)])]) == (False, False)
    assert find_support("42", "number", [("SELECT x FROM (SELECT 42 x), t", [(42,)])]) == (False, False)
    assert find_support("42", "number", [("SELECT v FROM t WHERE v = 42", [(42,)])]) == (True, True)
    assert find_support("42.004", "number", [("SELECT AVG(v) FROM t", [(42.0,)])]) == (True, False)
    assert find_support("Garden", "text", [("SELECT c FROM t", [("garden",), (None,)])]) == (True, False)
    assert find_support("42", "number", []) == (False, False)


def test_both_grpo_reward_inputs_block_the_literal_cheat(splits):
    from frugal_analyst.trajectory import format_tool_call_turn, make_grpo_reward

    task = first(splits, 1)
    query = f"SELECT {lit(task.gold_answer)} FROM {table_of(task)} LIMIT 1"
    call = ToolCall("run_sql", {"database": task.domain, "query": query})
    text = format_tool_call_turn([call]) + run_tool(call.name, call.arguments).output + f"\nANSWER: {task.gold_answer}"
    trajectory = {"steps": [{"action": {"tool_calls": [{"name": call.name, "arguments": call.arguments}]}}],
                  "final_response": f"ANSWER: {task.gold_answer}"}
    reward = make_grpo_reward("gated")
    assert reward([text], task_id=[task.id]) == [0.1]
    assert reward(["ignored"], task_id=[task.id], trajectories=[trajectory]) == [0.1]
