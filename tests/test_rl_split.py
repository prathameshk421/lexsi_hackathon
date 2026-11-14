"""The SFT / GRPO split of train: RL-only templates, train_sft and train_rl files."""

import json
import random
from pathlib import Path

import pytest

from frugal_analyst import tasks as T
from frugal_analyst.db import OOD_DOMAINS, TRAIN_DOMAINS, ensure_dbs
from frugal_analyst.traces import run_scripted_policy

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "data" / "tasks"
TEACHER_DIR = ROOT / "data" / "teacher"

ensure_dbs()


def load(name: str) -> list[T.Task]:
    path = TASKS_DIR / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} missing: run scripts/gen_data.py --seed 0")
    return T.load_tasks(path)


@pytest.fixture(scope="module")
def generated():
    return T.generate_tasks(0)


def test_rule_is_last_template_of_each_level():
    assert set(T.RL_ONLY_TEMPLATES) == set(TRAIN_DOMAINS)
    for domain in TRAIN_DOMAINS:
        names = T.RL_ONLY_TEMPLATES[domain]
        assert len(names) == 3
        by_name = {t.name: t for t in T.TEMPLATES[domain]}
        assert [by_name[n].level for n in names] == [1, 2, 3]
        for name in names:
            level = by_name[name].level
            later = [t for t in T.TEMPLATES[domain] if t.level == level]
            assert later[-1].name == name
    assert not any(d in T.RL_ONLY_TEMPLATES for d in OOD_DOMAINS)


def test_every_level_keeps_an_sft_template():
    # Each domain and level still has at least one template for SFT.
    for domain in TRAIN_DOMAINS:
        for level in (1, 2, 3):
            names = [t.name for t in T.TEMPLATES[domain] if t.level == level]
            assert len(names) >= 2
            assert any(n not in T.RL_ONLY_TEMPLATES[domain] for n in names)


def test_measured_counts():
    train, test_id = load("train"), load("test_id")
    train_sft, train_rl = load("train_sft"), load("train_rl")
    assert sum(len(v) for v in T.RL_ONLY_TEMPLATES.values()) == 12
    assert sum(T.is_rl_only(t) for t in train) == 126
    assert sum(T.is_rl_only(t) for t in test_id) == 38
    assert len(train_sft) == 400 - 126
    assert len(train_rl) == 126 + 126 // T.FORGET_CHECK_RATIO
    assert sum(T.is_rl_only(t) for t in train_rl) == 126


def test_saved_files_match_the_split_function():
    train_sft, train_rl = T.split_train_for_rl(load("train"))
    assert train_sft == load("train_sft")
    assert train_rl == load("train_rl")


def test_split_does_not_depend_on_input_order():
    train = load("train")
    shuffled = list(train)
    random.Random(7).shuffle(shuffled)
    assert T.split_train_for_rl(shuffled) == T.split_train_for_rl(train)


def test_train_sft_has_no_rl_only_template():
    for t in load("train_sft"):
        assert not T.is_rl_only(t), t.id
    sft_templates = {(t.domain, t.template) for t in load("train_sft")}
    for domain, names in T.RL_ONLY_TEMPLATES.items():
        for name in names:
            assert (domain, name) not in sft_templates


def test_train_rl_is_rl_only_plus_sft_seen_checks():
    train = {t.id: t for t in load("train")}
    sft_ids = {t.id for t in load("train_sft")}
    rl = load("train_rl")
    checks = [t for t in rl if not T.is_rl_only(t)]
    assert len(checks) == 42
    assert all(t.id in sft_ids for t in checks)
    assert {t.id for t in rl if T.is_rl_only(t)} == {i for i, t in train.items() if T.is_rl_only(t)}
    assert len({t.id for t in rl}) == len(rl)


def test_no_test_or_saas_task_in_any_train_file():
    train_ids = {t.id for t in load("train")}
    held_out = load("val") + load("test_id") + load("test_ood")
    held_ids = {t.id for t in held_out}
    held_questions = {t.question for t in held_out}
    teacher = TEACHER_DIR / "train_sft.jsonl"
    teacher_ids = [json.loads(line)["task_id"] for line in teacher.read_text().splitlines() if line.strip()]
    for name in ("train", "train_sft", "train_rl"):
        for t in load(name):
            assert t.split == "train" and t.domain not in OOD_DOMAINS, (name, t.id)
            assert t.id in train_ids and t.id not in held_ids, (name, t.id)
            assert t.question not in held_questions, (name, t.id)
    assert teacher_ids == [t.id for t in load("train_sft")]


def test_split_refuses_held_out_tasks():
    with pytest.raises(ValueError):
        T.split_train_for_rl(load("train") + load("test_id")[:1])
    with pytest.raises(ValueError):
        T.split_train_for_rl(load("test_ood")[:1])


def test_sft_teacher_lines_equal_the_full_teacher_lines():
    full = {}
    for line in (TEACHER_DIR / "train.jsonl").read_text().splitlines():
        full[json.loads(line)["task_id"]] = line
    for line in (TEACHER_DIR / "train_sft.jsonl").read_text().splitlines():
        assert full[json.loads(line)["task_id"]] == line


def test_files_are_byte_identical_to_a_fresh_run(generated, tmp_path):
    """gen_data.py must not change the old files, and the new files must be reproducible."""
    for split in T.SPLITS:
        T.save_tasks(generated[split], tmp_path / f"{split}.jsonl")
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (TASKS_DIR / f"{split}.jsonl").read_bytes(), split
    train_sft, train_rl = T.split_train_for_rl(generated["train"])
    T.save_tasks(train_sft, tmp_path / "train_sft.jsonl")
    T.save_tasks(train_rl, tmp_path / "train_rl.jsonl")
    for name in ("train_sft", "train_rl"):
        assert (tmp_path / f"{name}.jsonl").read_bytes() == (TASKS_DIR / f"{name}.jsonl").read_bytes(), name
    run_scripted_policy("oracle", train_sft, tmp_path / "teacher_sft.jsonl")
    assert (tmp_path / "teacher_sft.jsonl").read_bytes() == (TEACHER_DIR / "train_sft.jsonl").read_bytes()
