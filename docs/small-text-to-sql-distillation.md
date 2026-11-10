# Small text-to-SQL model via distillation (proposed plan)

Source: Aarush, Nov 10. Status: proposed candidate, not started. This doc records his idea, goals and caveats.

## Idea
Train a small text-to-SQL model with distillation that stays as capable as the big models.

## Goals
- Match big-model capability on text-to-SQL.
- Stretch: distill a model small enough to run on Aarush's laptop.

## Caveats
- Watch overfitting.
- Be careful about the learning signals used for distillation.

## Main obstacle
- Getting the distillation traces.

## Relation to the reconciliation v1
The reconciliation plan already uses distillation as a warm start before RL (see [docs/plan.md](plan.md)). This candidate is distillation as the main method for a standalone small text-to-SQL model, not the same thing.
