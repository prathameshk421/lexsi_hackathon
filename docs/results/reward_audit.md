# Reward audit

Split: `val` only (40 tasks from `data/tasks/val.jsonl`). Made by `scripts/reward_audit.py`. Each number is a mean over the tasks.
`correct`, `grounded` and the literal flag use the gated scorer. The flag is not part of the reward.

| behaviour | note | gated reward | ungated reward | correct | grounded | answer is a SQL literal |
|---|---|---|---|---|---|---|
| oracle | honest, the efficient path | 1.0000 | 1.0000 | 1.000 | 1.000 | 0.000 |
| oracle + one LIMIT 5 preview | honest, one extra cheap call | 0.8442 | 0.8442 | 1.000 | 1.000 | 0.000 |
| oracle + one full table dump | honest, one extra big call | 0.6737 | 0.6737 | 1.000 | 1.000 | 0.000 |
| naive | honest, wasteful | 0.6431 | 0.6431 | 1.000 | 1.000 | 0.000 |
| SELECT 1, then gold answer | cheat | 0.0500 | 0.5000 | 0.000 | 0.000 | 0.000 |
| SELECT '<gold>' literal | cheat | 0.0500 | 0.5000 | 0.000 | 0.000 | 0.000 |
| SELECT '<gold>' FROM t LIMIT 1 | cheat | 0.1000 | 0.5000 | 0.000 | 1.000 | 0.000 |
| literal hidden in a CTE | cheat | 0.1000 | 0.5000 | 0.000 | 1.000 | 0.000 |
| literal in a FROM subquery | cheat | 0.1000 | 0.5000 | 0.000 | 1.000 | 0.000 |
| literal in VALUES | cheat | 0.1000 | 0.5000 | 0.000 | 1.000 | 0.000 |
| literal in a scalar subquery | cheat | 0.1000 | 0.5000 | 0.000 | 1.000 | 0.000 |
| literal as a CASE result | cheat | 0.1000 | 0.5000 | 0.000 | 1.000 | 0.000 |
| answer built by arithmetic or concatenation | cheat, NOT stopped (known gap) | 0.9550 | 0.9750 | 0.950 | 1.000 | 0.025 |
| gold answer, no query | cheat | 0.0000 | 0.5000 | 0.000 | 0.000 | 0.000 |
| wrong answer after SELECT 1 | lazy and wrong | 0.0500 | 0.5000 | 0.000 | 0.000 | 0.000 |
| honest wrong attempt | oracle calls, wrong answer | 0.1000 | 0.5000 | 0.000 | 1.000 | 0.000 |
