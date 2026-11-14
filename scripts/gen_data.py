"""Build databases, task files, teacher trajectories and scripted test traces.

    python scripts/gen_data.py --seed 0

Same seed, same bytes in data/tasks/ and data/teacher/.

Besides the four splits it writes the SFT and GRPO subsets of train:
data/tasks/train_sft.jsonl and data/teacher/train_sft.jsonl (no RL-only
templates), and data/tasks/train_rl.jsonl (RL-only tasks plus forgetting checks).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python scripts/gen_data.py` finds frugal_analyst

from frugal_analyst.db import ensure_dbs  # noqa: E402
from frugal_analyst.tasks import (  # noqa: E402
    RL_ONLY_TEMPLATES,
    SPLITS,
    generate_tasks,
    is_rl_only,
    save_tasks,
    split_train_for_rl,
)
from frugal_analyst.traces import run_scripted_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db-dir", default=None, help="default: data/dbs, or FRUGAL_DB_DIR")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)

    ensure_dbs(args.db_dir, seed=args.seed, force=True)  # always rebuild so tasks and data match the seed
    splits = generate_tasks(args.seed, args.db_dir)

    for split in SPLITS:
        save_tasks(splits[split], data_dir / "tasks" / f"{split}.jsonl")
    run_scripted_policy("oracle", splits["train"], data_dir / "teacher" / "train.jsonl", args.db_dir)

    # SFT and GRPO subsets of train (see RL_ONLY_TEMPLATES in frugal_analyst/tasks.py).
    train_sft, train_rl = split_train_for_rl(splits["train"])
    save_tasks(train_sft, data_dir / "tasks" / "train_sft.jsonl")
    save_tasks(train_rl, data_dir / "tasks" / "train_rl.jsonl")
    run_scripted_policy("oracle", train_sft, data_dir / "teacher" / "train_sft.jsonl", args.db_dir)
    # Mixed teacher experiment: same tasks, half of them with one preview query that is not needed.
    mixed = run_scripted_policy("explorer", train_sft, data_dir / "teacher" / "train_sft_mixed.jsonl", args.db_dir)

    print("tasks per split:")
    for split in SPLITS:
        print(f"  {split:9s} {len(splits[split])}")
    n_rl_only = sum(is_rl_only(t) for t in train_rl)
    print(f"  train_sft {len(train_sft)} (train minus RL-only templates)")
    print(f"  train_rl  {len(train_rl)} ({n_rl_only} RL-only + {len(train_rl) - n_rl_only} SFT-seen forgetting checks)")
    n_mixed = len(mixed)
    print(f"  mixed teacher: {sum(t['cost']['tool_calls'] > t['optimal']['tool_calls'] for t in mixed)} of {n_mixed} traces have a preview, "
          f"correct={sum(t['correct'] for t in mixed) / n_mixed:.3f} mean_reward={sum(t['reward'] for t in mixed) / n_mixed:.3f} "
          f"max_calls={max(t['cost']['tool_calls'] for t in mixed)}")
    print(f"RL-only templates: {sum(len(v) for v in RL_ONLY_TEMPLATES.values())}, "
          f"test_id tasks from them: {sum(is_rl_only(t) for t in splits['test_id'])}")
    print("tasks per split, domain and level:")
    for split in SPLITS:
        counts = Counter((t.domain, t.level) for t in splits[split])
        for (domain, level), n in sorted(counts.items()):
            print(f"  {split:9s} {domain:10s} L{level}  {n}")

    print("scripted traces for the test splits:")
    for split in ("test_id", "test_ood"):
        for policy in ("oracle", "naive"):
            traces = run_scripted_policy(policy, splits[split], out_dir / "traces" / f"{policy}_{split}.jsonl", args.db_dir)
            n = len(traces)
            print(
                f"  {policy:7s} {split:9s} n={n} correct={sum(t['correct'] for t in traces) / n:.3f} "
                f"mean_efficiency={sum(t['efficiency'] for t in traces) / n:.3f} "
                f"mean_reward={sum(t['reward'] for t in traces) / n:.3f}"
            )


if __name__ == "__main__":
    main()
