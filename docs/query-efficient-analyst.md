# Query-efficient analyst (proposed plan)

Source: Aarush, Nov 10. Status: proposed candidate, not started. This doc records his idea, hypothesis and caveats.

## Idea
An analyst agent like the Databricks agent: RL-train the agent on query optimization paths, so it learns to answer questions with efficient SQL rather than only correct SQL.

## Hypothesis (Aarush)
This gets tricky because most of the optimization part in SQL is already handled by the database engine. How much room the agent's optimization-path behavior has to matter is an open question.

## Caveats
- Be careful about reward hacking: an efficiency reward can be gamed (for example by avoiding needed work) instead of genuinely optimizing.

## Open questions / rough next steps
- Pin down what the environment measures: latency, rows scanned, or cost estimates from the engine.
- Define a reward that resists gaming (correctness gate plus efficiency score).
- Check what the database engine already optimizes away, so the agent's contribution is real and measurable.
