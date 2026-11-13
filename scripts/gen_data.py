"""Build databases, task files, teacher trajectories and scripted test traces.

    python scripts/gen_data.py --seed 0

Same seed, same bytes in data/tasks/ and data/teacher/.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python scripts/gen_data.py` finds frugal_analyst

from frugal_analyst.db import ensure_dbs  # noqa: E402
from frugal_analyst.tasks import SPLITS, generate_tasks, save_tasks  # noqa: E402
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

    print("tasks per split:")
    for split in SPLITS:
        print(f"  {split:9s} {len(splits[split])}")
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
