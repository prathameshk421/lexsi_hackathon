import re
from collections import Counter
from dataclasses import asdict

import pytest

from frugal_analyst import tasks as T
from frugal_analyst.db import DOMAINS, OOD_DOMAINS, TRAIN_DOMAINS, ensure_dbs
from frugal_analyst.env import execute_metered, format_number

ensure_dbs()


@pytest.fixture(scope="module")
def splits():
    return T.generate_tasks(0)


@pytest.fixture(scope="module")
def all_tasks(splits):
    return [t for tasks in splits.values() for t in tasks]


def test_generation_is_deterministic(tmp_path):
    a = T.generate_tasks(0, tmp_path / "a")
    b = T.generate_tasks(0, tmp_path / "b")
    assert {s: [asdict(t) for t in ts] for s, ts in a.items()} == {s: [asdict(t) for t in ts] for s, ts in b.items()}


def test_different_seed_changes_tasks(tmp_path, splits):
    other = T.generate_tasks(1, tmp_path)
    assert [asdict(t) for t in other["test_id"]] != [asdict(t) for t in splits["test_id"]]


def test_split_sizes(splits):
    assert set(splits) == {"train", "val", "test_id", "test_ood"}
    assert len(splits["train"]) == 400
    assert len(splits["val"]) == 40
    assert len(splits["test_id"]) == 120
    assert len(splits["test_ood"]) == 100


def test_ood_split_is_only_the_held_out_domain(splits):
    assert {t.domain for t in splits["test_ood"]} == set(OOD_DOMAINS)
    for split in ("train", "val", "test_id"):
        assert {t.domain for t in splits[split]} == set(TRAIN_DOMAINS)


def test_every_split_has_every_level_and_split_label_matches(splits):
    for split, ts in splits.items():
        assert {t.level for t in ts} == {1, 2, 3}
        assert all(t.split == split for t in ts)


def test_no_question_is_shared_between_splits(splits):
    owner = {}
    for split, ts in splits.items():
        for t in ts:
            assert owner.setdefault(t.question, split) == split, t.question
    assert len(owner) == sum(len(ts) for ts in splits.values())  # not even repeated inside a split


def test_no_template_parameter_pair_repeats(all_tasks):
    # Every parameter appears in the gold SQL, so (domain, template, gold_sql) identifies the pair.
    keys = [(t.domain, t.template, t.gold_sql) for t in all_tasks]
    assert len(keys) == len(set(keys))


def test_ids_are_unique_and_well_formed(all_tasks):
    assert len({t.id for t in all_tasks}) == len(all_tasks)
    for t in all_tasks:
        assert re.fullmatch(rf"{t.domain}-L{t.level}-\d{{4}}", t.id)


def test_at_least_eight_templates_per_domain_and_three_levels():
    for domain in DOMAINS:
        templates = T.TEMPLATES[domain]
        assert len(templates) >= 8
        assert {t.level for t in templates} == {1, 2, 3}
        assert len({t.name for t in templates}) == len(templates)


def test_questions_vary(all_tasks):
    per_template = {}
    for t in all_tasks:
        per_template.setdefault((t.domain, t.template), []).append(t.question)
    for (domain, template), questions in per_template.items():
        tpl = next(x for x in T.TEMPLATES[domain] if x.name == template)
        if len(tpl.questions) > 1 and len(questions) >= 10:
            # With several phrasings, 10+ questions must not all start the same way.
            assert len({q.split()[0] + q.split()[1] for q in questions}) > 1 or len({q[-12:] for q in questions}) > 1


def test_oracle_call_shape(all_tasks):
    for t in all_tasks:
        assert t.oracle_calls[0] == {"name": "get_schema", "arguments": {"database": t.domain}}
        assert t.oracle_calls[-1] == {"name": "run_sql", "arguments": {"database": t.domain, "query": t.gold_sql}}
        assert t.optimal_tool_calls == len(t.oracle_calls) == (3 if t.level == 3 else 2)
        assert t.optimal_rows >= 1 and t.optimal_vm_steps > 0


def test_gold_answers_reproduce_and_are_usable(all_tasks):
    for t in all_tasks:
        res = execute_metered(t.domain, t.gold_sql)
        assert res.ok and res.n_rows == 1 and len(res.rows[0]) == 1, t.id
        value = res.rows[0][0]
        assert value is not None, t.id  # numeric answers are never NULL
        if t.answer_type == "number":
            assert isinstance(value, (int, float)) and t.gold_answer == format_number(value), t.id
            float(t.gold_answer)
        else:
            assert t.answer_type == "text" and t.gold_answer == value and value.strip() == value and value, t.id


def test_oracle_cost_fields_match_a_rerun(all_tasks):
    for t in all_tasks[::25]:
        sql_calls = [c for c in t.oracle_calls if c["name"] == "run_sql"]
        results = [execute_metered(t.domain, c["arguments"]["query"]) for c in sql_calls]
        assert t.optimal_rows == sum(r.n_rows for r in results)
        assert t.optimal_vm_steps == sum(r.vm_steps for r in results)


def test_text_answers_are_unique_top_one(all_tasks):
    """Re-check the tie rule from outside the generator, using each task's own template."""
    checked = 0
    for t in all_tasks:
        tpl = next(x for x in T.TEMPLATES[t.domain] if x.name == t.template)
        if tpl.tie_sql is None:
            continue
        # Turn the gold SQL back into a LIMIT 2 query by swapping the select list.
        limit2 = t.gold_sql.replace("LIMIT 1", "LIMIT 2")
        key_end = limit2.index(" FROM ")
        metric = tpl.tie_sql.split(" FROM ")[0].removeprefix("SELECT ")
        res = execute_metered(t.domain, f"SELECT {metric}" + limit2[key_end:])
        assert res.ok, t.id
        if res.n_rows == 2:
            assert round(res.rows[0][0], 6) != round(res.rows[1][0], 6), t.id
        checked += 1
    assert checked > 50


def test_text_templates_all_have_a_tie_check():
    for domain in DOMAINS:
        for tpl in T.TEMPLATES[domain]:
            assert (tpl.tie_sql is not None) == (tpl.answer_type == "text" and "LIMIT 1" in tpl.sql)


def test_level3_really_needs_a_lookup(all_tasks):
    level3 = [t for t in all_tasks if t.level == 3]
    assert len(level3) > 100
    for t in level3:
        lookup = t.oracle_calls[1]["arguments"]["query"]
        shown = execute_metered(t.domain, lookup)
        assert shown.ok and shown.n_rows >= 2
        # A code is used in the gold SQL, it is absent from the question, and the lookup reveals it.
        literals = [lit for lit in re.findall(r"'([^']+)'", t.gold_sql) if not re.fullmatch(r"[\d-]+", lit)]
        coded = [lit for lit in literals if any(lit == str(v) for row in shown.rows for v in row)]
        assert coded, t.id
        for code in coded:
            assert not re.search(rf"\b{re.escape(code)}\b", t.question, re.IGNORECASE), (t.id, code)


def test_levels_1_and_2_need_no_lookup(all_tasks):
    for t in all_tasks:
        if t.level < 3:
            assert len(t.oracle_calls) == 2


def test_jsonl_round_trip(tmp_path, splits):
    path = tmp_path / "x" / "train.jsonl"
    T.save_tasks(splits["train"], path)
    assert T.load_tasks(path) == splits["train"]
    assert len(path.read_text().splitlines()) == 400


def test_balance_across_templates(all_tasks):
    counts = Counter((t.domain, t.template) for t in all_tasks)
    for (domain, _), n in counts.items():
        assert n >= (8 if domain in OOD_DOMAINS else 12)
