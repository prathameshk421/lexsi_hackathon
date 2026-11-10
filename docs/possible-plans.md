# Possible plans - Lexsi hackathon, Agentic Fine-tuning track

This is the main overview of the plans on the table for team Dear_Basketball. It describes each candidate idea, its status, and links to its own plan doc.

## The three ideas

### 1. Reconciliation agent (current v1)
- **What:** Compare two systems that should agree, find real discrepancies, and ignore harmless rounding noise. PS6 with PS9-informed reward design.
- **Status:** Active v1 plan. This is what the team is building now.
- **Plan:** [docs/plan.md](plan.md)

### 2. Query-efficient analyst (proposed)
- **What:** An analyst agent in the style of the Databricks agent, RL-trained on query optimization paths.
- **Status:** Proposed candidate from Aarush (Nov 10). Not started. Aarush wants the team to prioritise evaluating it.
- **Plan:** [docs/query-efficient-analyst.md](query-efficient-analyst.md)

### 3. Small text-to-SQL model via distillation (proposed)
- **What:** Distill a small model that stays as capable as the big models, ideally small enough to run on a laptop.
- **Status:** Proposed candidate from Aarush (Nov 10). Not started. Aarush wants the team to prioritise evaluating it.
- **Plan:** [docs/small-text-to-sql-distillation.md](small-text-to-sql-distillation.md)

Aarush's read (Nov 10): most teams will build generic text-to-sql agents, so these angles are worth looking into. Ideas 2 and 3 are candidates to evaluate, not replacements for the reconciliation v1; nothing here is implemented yet.
