# Frugal Analyst: build spec

This is the contract for all code in this repo. If code and spec disagree, fix one of them in the same change.

## One-line idea

Train Tiny Aya (`CohereLabs/tiny-aya-global`, 3.35B parameters) to answer analytics questions over SQLite with tools, and to do it cheaply: few tool calls, few rows pulled into context, low engine work.

It is Aarush's "query-efficient analyst" plan. Track 3, problem statements 2 (efficient trajectories), 6 (database agents), 3 (agentic distillation) and 9 (reward-hacking defences).

## Why efficiency is real (answer to Aarush's open question)

The database engine picks the plan for one query. It cannot fix what the agent does around the query:

1. **Round trips.** Each tool call is one more model turn.
2. **Rows in context.** `SELECT *` and "look at the rows" costs tokens. An aggregate in SQL costs one row.
3. **Query shape.** A missing filter or a needless join makes the engine do more work. SQLite reports this as VM instructions (`sqlite3` progress handler). The count is deterministic, so it is a fair cost metric. Latency is not.

## Pipeline

1. `zero-shot`: Tiny Aya with tools, no training. This is "before".
2. `distilled`: LoRA SFT on scripted teacher trajectories (AgentTune `Project.distill`). Warm start.
3. `grpo`: GRPO with tools on top of the distilled adapter (AgentTune `create_agentic_trainer("grpo", tools=...)`). Reward = correctness gate times efficiency.
4. Optional ablation, only if budget allows: GRPO with an ungated efficiency reward, to show reward hacking (the model stops querying), then show the gate fixes it.
5. Evaluation with AuditKit on in-distribution and held-out-domain test sets. Paired bootstrap for before/after.

## Layout

```
frugal_analyst/
  __init__.py
  db.py          synthetic SQLite databases, one per domain, seeded
  tasks.py       Task dataclass, templated task generator, split logic, jsonl io
  env.py         tools (get_schema, run_sql), read-only metered SQL execution
  prompts.py     SYSTEM_PROMPT, build_user_prompt(task)
  reward.py      score_trajectory(), answer matching, RewardConfig
  policies.py    scripted policies: oracle (teacher), naive (wasteful), used with no GPU
  traces.py      trace record schema, jsonl read/write, run_scripted_policy()
  metrics.py     per-episode metrics, aggregate(), paired bootstrap interval
  trajectory.py  parse AgentTune completion text into tool calls + final answer   [training agent]
  runner.py      run a HF model (+ optional LoRA adapter) over tasks, write traces [training agent]
scripts/
  gen_data.py        build dbs, tasks, teacher trajectories
  train_sft.py       distillation with AgentTune                                   [training agent]
  train_grpo.py      GRPO with AgentTune                                           [training agent]
  eval_auditkit.py   AuditKit evaluation over trace files                          [eval agent]
  eval_agent.py      AuditKit agent evaluator: tool-call accuracy vs oracle calls  [eval agent]
  rescore_traces.py  replay old trace files and score them with the current reward [eval agent]
  reward_audit.py    mean reward of honest and cheating behaviours on val           [eval agent]
  jobs/              HF Jobs entry points                                          [training agent]
app/
  app.py             Gradio demo, also the HF Space entry point                    [demo agent]
tests/
data/tasks/*.jsonl   generated, committed (small)
data/teacher/*.jsonl generated, committed (small)
data/dbs/*.sqlite    generated, git-ignored, rebuilt by gen_data.py
outputs/             runs, traces, eval results. git-ignored
```

Python 3.12. The venv is `.venv/` at the repo root. Run tools as `.venv/bin/python`.
The core package (`db`, `tasks`, `env`, `prompts`, `reward`, `policies`, `traces`) uses the standard library only.

## Databases (`db.py`)

- `DOMAINS = ["retail", "clinic", "logistics", "school", "saas"]`. Train domains are the first four. `saas` is held out (OOD test only).
- Each domain: 3 or 4 tables with foreign keys, 300 to 3000 rows per table, realistic column names, dates as ISO text.
- Each domain has at least two "coded" columns where the stored value is not guessable from the question, for example `status = 'CXL_USR'` for "cancelled by the customer". A small lookup table or a `SELECT DISTINCT` reveals the codes. Level 3 tasks need this.
- `build_db(domain: str, path: str | Path, seed: int = 0) -> None`. Deterministic for a given seed. Same bytes of data each time.
- `db_path(domain: str, db_dir: str | Path | None = None) -> Path`. Default dir is `data/dbs/`. Env var `FRUGAL_DB_DIR` overrides it.
- `ensure_dbs(db_dir=None) -> None` builds any missing database.
- `schema_text(domain: str, db_dir=None) -> str`: compact schema, one line per table: `table(col type, col type, ...)` plus row count. No sample values.

## Tools and metering (`env.py`)

Two tools. They are plain functions with type hints and Google-style docstrings, because AgentTune builds the JSON schema from them.

```python
def get_schema(database: str) -> str: ...
def run_sql(database: str, query: str) -> str: ...
TOOLS = [get_schema, run_sql]
```

- Tools are stateless. The database name comes from the prompt. Unknown database returns `ERROR: unknown database ...` and lists valid names.
- `run_sql` opens the file read-only (`file:...?mode=ro`, `uri=True`). Only one statement. Only `SELECT` or `WITH`. Anything else returns `ERROR: only SELECT queries are allowed`.
- Output format: header line with column names joined by ` | `, then rows. At most `MAX_ROWS_SHOWN = 20` rows, then a line `... (N more rows not shown)`. Output is cut at `MAX_OUTPUT_CHARS = 1500`.
- Errors are returned as text that starts with `ERROR:`. Tools never raise.
- Metering: `execute_metered(database, query, db_dir=None) -> SqlResult` with fields `ok: bool, columns: list[str], rows: list[tuple], n_rows: int, vm_steps: int, error: str | None, tables_read: tuple[str, ...]`. VM steps come from `set_progress_handler` with a small period (`PROGRESS_PERIOD = 10`) times the callback count. A query that passes `MAX_VM_STEPS = 5_000_000` is aborted and returns an error. `n_rows` is the full row count of the result, fetched up to `MAX_ROWS_FETCHED = 2000`.
- `tables_read`: the data tables the query read, sorted, no repeats. The SQLite authorizer records the table name of each `SQLITE_READ` while SQLite compiles the query. The allow/deny rule is the same as before. `SELECT COUNT(*) FROM orders` gives `("orders",)`. `SELECT 1` and a CTE that holds only literals give `()`. Tables whose name starts with `sqlite_` hold the schema, so they do not count. A failed query gives `()`.
- VM-step counts are not portable between machines. The `zero-shot` and `distilled` traces in `outputs/traces/` came from model runs on another machine. A local replay (SQLite 3.54.1) gives other VM steps for 13 of the 880 recorded traces, all join or lookup queries. Rows and output text are the same. The likely cause is a different SQLite build. This is not checked.
- `render_result(result: SqlResult) -> str` gives the tool output text. `run_sql` is `render_result(execute_metered(...))`.
- `run_tool(name, arguments, db_dir=None) -> ToolResult` with fields `output, ok, n_rows, vm_steps, is_sql, rows, tables_read`. `rows` holds every fetched row, not only the 20 rows shown.

## Tasks (`tasks.py`)

```python
@dataclass
class Task:
    id: str                 # stable, e.g. "retail-L2-0042"
    domain: str
    level: int              # 1, 2 or 3
    template: str           # template name
    question: str
    answer_type: str        # "number" or "text"
    gold_sql: str           # final query of the efficient solution
    gold_answer: str        # canonical string
    oracle_calls: list[dict]  # efficient tool-call sequence: [{"name": ..., "arguments": {...}}, ...]
    optimal_tool_calls: int   # len(oracle_calls)
    optimal_rows: int         # total rows returned across oracle run_sql calls
    optimal_vm_steps: int     # total vm steps across oracle run_sql calls
    split: str              # "train" | "val" | "test_id" | "test_ood"
```

- Level 1: one table, filter plus aggregate. Level 2: join plus aggregate or group-by with top-1. Level 3: needs a coded value, so the efficient path is schema, one cheap lookup (`SELECT DISTINCT ...` or the lookup table), then the query.
- The efficient path always starts with `get_schema`. So `optimal_tool_calls` is 2 for levels 1 and 2, and 3 for level 3.
- Every answer is one scalar: a number (count, sum, average, max, min) or one text value (a name, a category). No list answers. No yes/no answers. Numbers are rounded to 2 decimals in `gold_answer`. Avoid ties for top-1 questions: the generator must check the gold result is unique and drop the task if not.
- At least 8 templates per domain. Questions use natural wording with some variety, not the column names verbatim for coded values.
- Splits: `train` about 400, `val` about 40, `test_id` about 120 (train domains), `test_ood` about 100 (`saas` only). No two tasks share the same `(template, parameters)`. No `question` string appears in more than one split.
- `generate_tasks(seed: int = 0, db_dir=None) -> dict[str, list[Task]]`, `save_tasks(tasks, path)`, `load_tasks(path) -> list[Task]`.
- The generator runs every `gold_sql` and every oracle call through `execute_metered` to fill `gold_answer` and the `optimal_*` fields. A task whose oracle path fails is a bug.

## Prompts (`prompts.py`)

- `SYSTEM_PROMPT`: short. It says: you are a data analyst; use `get_schema` and `run_sql`; every call costs, so use as few calls as possible, let SQL do the aggregation, and never fetch rows you do not need; end with one line `ANSWER: <value>`.
- `build_user_prompt(task) -> str` returns `Database: {domain}\nQuestion: {question}`.
- The final answer contract is the literal prefix `ANSWER:` on the last non-empty line.

## Reward (`reward.py`)

```python
@dataclass
class ToolCall: name: str; arguments: dict
@dataclass
class EpisodeCost: tool_calls: int; sql_calls: int; failed_calls: int; rows_returned: int; vm_steps: int
@dataclass
class Score:
    answered: bool; grounded: bool; correct: bool
    efficiency: float      # 0..1, defined even when incorrect (for analysis)
    reward: float
    cost: EpisodeCost
    predicted: str | None
    supported: bool = False
    answer_in_sql_literal: bool = False

def extract_answer(text: str | None) -> str | None
def answers_match(predicted: str, gold: str, answer_type: str) -> bool
def sql_literals(query: str) -> list[str]
def select_list_literals(query: str) -> list[str]
def find_support(predicted: str, answer_type: str, evidence: list[tuple[str, list[tuple]]]) -> tuple[bool, bool]
def score_trajectory(task: Task, calls: list[ToolCall], final_text: str | None,
                     cfg: RewardConfig = RewardConfig(), db_dir=None) -> Score
```

- `score_trajectory` re-executes every tool call with `run_tool` to get rows, VM steps and `tables_read`. It does not trust tool output text. This keeps the reward stateless and deterministic. Both GRPO inputs (the `trajectories` kwarg and the text fallback) become a list of calls first, so both use this path.
- `grounded`: at least one successful `run_sql` call read at least one data table (`tables_read` is not empty). `SELECT 1` and `SELECT '42'` are not grounding.
- `supported`: some successful, table-reading `run_sql` result has a cell that matches the predicted answer (same rule and tolerance as `answers_match`, with the cell as the reference value), and the predicted answer is not a literal in any select list of that query, at any nesting depth.
  - A select list is the text after a `SELECT` (or `VALUES`) up to the next `FROM`, `WHERE`, `GROUP`, `HAVING`, `WINDOW`, `ORDER`, `LIMIT`, `UNION`, `INTERSECT` or `EXCEPT` at the same parenthesis depth, or up to the `)` that closes its subquery. `select_list_literals` tracks one "inside a select list" flag per parenthesis depth. A `(` inside a select list (a function call, a `CASE`, a scalar subquery) starts inside one too, and a `(` in `FROM` or `WHERE` starts outside one.
  - So these count: CTE bodies, `FROM` subqueries, scalar subqueries, every `UNION` part, `VALUES` rows, a `SELECT` with no `FROM`, function arguments and `CASE` branches (conditions and results) in a select list, and the select list of a subquery inside `WHERE`. Comments are removed first.
  - These do not count, on purpose: `WHERE`, `ON`, `HAVING`, `GROUP BY`, `ORDER BY`, `LIMIT`, and `IN (...)` lists. A real filter can hold a number that equals the answer: 2 of 400 train tasks (`school-L3-0012`, `school-L3-0039`) filter on `grade_level = 11` or `10`, and their oracle path scores 1.0.
  - False positives: 0. The oracle and naive paths of all 660 tasks (all four splits) stay supported and correct. No gold query has the answer as a select-list literal. The test splits were checked once as a sanity check, not for tuning.
  - Cells come from every row the engine fetched (up to `MAX_ROWS_FETCHED`), not only the 20 rows shown or the text left after the output cap. The rule then does not depend on display limits. The cost: a large dump can hold the value by chance. The answer must still equal gold, and the dump costs efficiency.
  - Known gaps. (1) An answer computed from other literals passes: `SELECT (360.0 / 2) FROM orders LIMIT 1` or `SELECT 'Gar' || 'den' FROM products LIMIT 1`. (2) A filter value echoed back through a column passes, because it looks the same as the 2 honest tasks: `SELECT grade_level FROM students WHERE grade_level = 10 LIMIT 1`. The flag below reports (2), not (1).
- `answer_in_sql_literal`: a suspicion flag, not part of the reward. It is true when the answer is supported, but every supporting query also has the predicted value as a literal outside its select lists (for example in `WHERE`). On the oracle path it is true for the 2 train tasks above and for no val or test task.
- `correct`: answered, supported, and `answers_match` with gold. Number match: parse both as float after removing commas, currency signs and a trailing `%`; equal if `abs(a-b) <= 0.01` or relative error `<= 0.001`. Text match: case-insensitive, trimmed, surrounding quotes and final period removed.
- Efficiency is the mean of three parts, each in 0..1:
  - calls: `min(1, optimal_tool_calls / tool_calls)`
  - rows: `min(1, (optimal_rows + 1) / (rows_returned + 1))`
  - engine: `min(1, (optimal_vm_steps + 1) / (vm_steps + 1))`
  With zero tool calls, efficiency is 0.
- Gated reward (default, `RewardConfig(mode="gated")`):
  - correct: `0.5 + 0.5 * efficiency`
  - not correct but grounded and answered: `0.1`
  - at least one valid tool call (any tool, no error) but no grounded answer: `0.05`
  - else `0.0`
- Ungated reward for the ablation (`mode="ungated"`): `0.5 * correct + 0.5 * efficiency_naive`, where `efficiency_naive` gives full marks for zero calls. This is the hackable reward. It must be easy to see in tests that "answer with no query" beats an honest wrong attempt under this mode and loses under the gated mode. `correct` uses the same evidence rule in both modes.
- Required tests: oracle scores 1.0 on every train and val task; empty trajectory scores 0.0; correct answer with a `SELECT *` dump scores below the oracle; a correct number with no query scores at most 0.05 in gated mode (not grounded); `SELECT 1` then gold scores at most 0.05; `SELECT '<gold>'` scores at most 0.05; `SELECT '<gold>' FROM t LIMIT 1` scores at most 0.1; `SELECT 1` then a wrong answer scores 0.05; oracle plus one `LIMIT 5` preview scores strictly between naive and oracle; the 2 tasks with the answer in `WHERE` still score 1.0; a literal in a CTE, a `FROM` subquery, `VALUES`, a scalar subquery or a `CASE` result scores at most 0.1; the `WHERE` echo gap is tested and documented.
- `scripts/reward_audit.py` scores scripted honest and cheating behaviours on the val split only (never a test split), under both modes. It prints a markdown table and writes `outputs/eval/reward_audit.md`.

## Scripted policies (`policies.py`)

No model. Each returns `(calls: list[ToolCall], final_text: str)`.
- `oracle_policy(task)`: the efficient path from `task.oracle_calls`, final text `ANSWER: {gold_answer}`.
- `naive_policy(task)`: correct but wasteful. It calls `get_schema`, runs `SELECT *` on each table the gold query touches, maybe one failed query, then the gold query. It models an untrained agent that is right but expensive.
- `explorer_policy(task)`: the oracle path plus one `SELECT * FROM <first table> LIMIT 5` after `get_schema`, on a stable half of the tasks (crc32 of the task id). It is the "mixed teacher" experiment. We add this waste on purpose, so GRPO has a habit to remove. The write-up must say so.
- `guess_policy(task)`: no tool calls, just a made-up answer. Used in reward tests.

## Traces (`traces.py`)

One JSON object per line. This is the only interface between runs, evaluation and the demo.

```json
{
  "task_id": "retail-L2-0042", "domain": "retail", "level": 2, "split": "test_id",
  "question": "...", "gold_answer": "...", "answer_type": "number",
  "model": "zero-shot",
  "steps": [
    {"type": "tool_call", "name": "run_sql", "arguments": {"database": "retail", "query": "..."},
     "ok": true, "n_rows": 1, "vm_steps": 812, "output": "..."},
    {"type": "final", "text": "ANSWER: 42"}
  ],
  "predicted": "42", "answered": true, "grounded": true, "supported": true, "correct": true,
  "answer_in_sql_literal": false, "efficiency": 1.0, "reward": 1.0,
  "cost": {"tool_calls": 2, "sql_calls": 1, "failed_calls": 0, "rows_returned": 1, "vm_steps": 812},
  "optimal": {"tool_calls": 2, "rows": 1, "vm_steps": 812},
  "raw_completion": null, "completion_tokens": null, "latency_s": null
}
```

- `build_trace(task, calls, final_text, model_label, raw_completion=None, completion_tokens=None, latency_s=None, db_dir=None) -> dict` executes the calls, scores, and fills every field.
- `supported` and `answer_in_sql_literal` came with the evidence rule. Trace files written before it do not have them. Readers use `.get()`, and the validator accepts a trace without them.
- `write_traces(path, traces)`, `read_traces(path) -> list[dict]`.
- `scripts/rescore_traces.py --traces ... --out DIR` scores old trace files again with the current reward. It replays every recorded tool call against the local databases and stops, writing nothing, if a replayed output, ok flag or row count differs. A VM-step difference also stops it, unless `--allow-vm-drift` is given. It never writes into the folder of an input file. It prints how many traces changed `correct`, `grounded` or `reward`.
- `run_scripted_policy(policy_name, tasks, out_path)` for `oracle`, `naive`, `guess`.
- Model labels used everywhere: `zero-shot`, `distilled`, `grpo`, `grpo-ungated`, `oracle`, `naive`.

## `scripts/gen_data.py`

`python scripts/gen_data.py --seed 0` builds the databases, writes `data/tasks/{train,val,test_id,test_ood}.jsonl`, writes `data/teacher/train.jsonl` (oracle traces for the train split), and writes scripted traces for both test splits to `outputs/traces/{oracle,naive}_{test_id,test_ood}.jsonl`. It prints counts per split, domain and level. Running it twice gives identical task files.

It also writes `data/tasks/train_sft.jsonl` and `data/teacher/train_sft.jsonl` (train tasks without RL-only templates), `data/teacher/train_sft_mixed.jsonl` (the same tasks with `explorer_policy`), and `data/tasks/train_rl.jsonl` (all RL-only train tasks plus one SFT-seen train task per three RL-only tasks, chosen by the lowest crc32 of the task id). RL-only templates: in each train domain, the last template of each level in `TEMPLATES` order (`RL_ONLY_TEMPLATES`, `is_rl_only` in `tasks.py`). No val, test or saas task enters any train file.

## Training and inference (training agent)

- Base model: `CohereLabs/tiny-aya-global`. It is gated; the HF token on this machine has access.
- AgentTune is installed from source at a pinned commit. The exact message and tool-call format that AgentTune renders for Tiny Aya must be read from AgentTune's source, not guessed. SFT rows must use the same format that the GRPO rollout and `runner.py` use at inference.
- `trajectory.py`: `parse_completion(text: str) -> tuple[list[ToolCall], str | None]` for AgentTune's flattened completion text. Per AgentTune's own docs, agentic GRPO passes each completion to the reward function as a flat string, not a message list.
- GRPO reward function: parse, then `score_trajectory`, return `score.reward`. Dataset columns `task_id` etc. arrive as kwargs.
- `runner.py`: `run_model(model_id, adapter, tasks, out_path, label, ...)` writes trace files in the schema above. It must use the same prompt rendering and tool loop as training.
- SFT trains on `data/teacher/train_sft.jsonl`. GRPO trains on `data/tasks/train_rl.jsonl` from the SFT adapter. The first run used all 400 train tasks for both, and GRPO had no reward spread. It is kept as the reference `distilled-all`.
- `--max-tool-steps` is the number of executed tool calls per episode. An answer turn always follows. The default is 4 in SFT, GRPO and the runner.
- The runner ends a model turn when its text ends with a complete tool-call list (`trajectory.ends_with_tool_call`). `--no-stop-after-tool-call` turns this off. The harness settings go to `<trace>.run.json`.
- `train_grpo.py` stops with exit code 4 if the first N steps (`--stop-if-no-signal`, default 3) have no reward spread in any group.
- All GPU work runs on HF Jobs. Total budget is 20 USD. Every script takes `--max-steps` or `--limit` so a smoke run costs cents.

## Evaluation (eval agent)

`scripts/eval_auditkit.py --traces outputs/traces/*.jsonl --out outputs/eval/` uses AuditKit (`import auditkit as ak`) with custom `@ak.scorer` functions over precomputed outputs. Metrics per model and split: accuracy, grounded rate, answered rate, "answered with no table read" rate (lower is better), the `answer_in_sql_literal` rate (lower is better, only for traces that have the field), mean reward, mean efficiency, mean tool calls, mean rows returned, mean VM steps, VM-step ratio to optimal, and "correct and optimal" rate. It runs one `ak.evaluate` per model and uses `ak.compare` for significance. `compare_models` does not work with precomputed outputs (see `docs/eval-notes.md`). The 95% interval is our own paired bootstrap in `frugal_analyst/metrics.py`. It writes `results.json` and `results.md`, and reports AuditKit's failed-sample count. It fails loudly if a trace file is malformed.

`scripts/eval_agent.py --traces outputs/traces/*.jsonl --out outputs/eval/` gives the tool-call accuracy metric. It uses `auditkit.agent_eval` (`AgentEvalRunner`, recorded mode). The reference for each task is `oracle_calls` from `data/tasks/<split>.jsonl`, never the steps of the trace. The headline metric is the name-level F1 of tool calls (higher is better). The script also reports name precision and recall, argument-level F1, repeated call share (lower is better), loop-free rate, schema-valid call rate and task success. It breaks in-domain results down by template group, compares each model with the baseline (default `zero-shot`) using our paired bootstrap, and writes `agent_results.json`, `agent_results.md` and the raw AuditKit runs in `agent_runs/`. Exit codes: 0 ok, 1 failed or errored episodes, 2 bad input, 3 AuditKit and the plain-Python recomputation of name-level and argument-level precision, recall and F1 differ by more than 1e-6. Definitions are in `docs/eval-notes.md`.

## Demo (demo agent)

`app/app.py` is a Gradio app. It must work with no GPU from trace files and the scripted policies.
- Replay tab: pick a split and a task, then see two models side by side, step by step: tool calls, outputs, cost meters (calls, rows, VM steps), correct or not.
- Live tab: pick a database, type or pick a question, run a policy. Scripted policies always work. A real model option appears only if the model can be loaded.
- SQL playground: type a query, see the result and its VM-step cost.
- Results tab: the table and charts from `outputs/eval/results.json` if it exists, and a clear "no results yet" state if not.

## Rules for every agent

- Do not invent APIs. Check the installed package or its source before you call it.
- Do not write metric numbers by hand anywhere. Numbers come from runs.
- Stay inside the files assigned to you. Do not edit files owned by another agent; report the need instead.
- No git commits. No network publishing. No HF Jobs unless your task says so.
- Tests must pass before you report done. Report failures as they are.
