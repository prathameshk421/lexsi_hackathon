"""Reward: correctness gates the efficiency bonus.

`score_trajectory` runs every tool call again against the databases. It never
trusts the tool output text a model saw. This keeps the reward stateless and
deterministic, and it means a model cannot fake a tool result.

Correctness needs evidence. The answer must match gold, and it must also appear
as a cell in the result of a query that read a real data table. The value must
not be a literal in any select list of that query, at any depth (CTEs,
subqueries, UNION parts, VALUES, CASE results). So
`SELECT '42' FROM orders LIMIT 1` followed by `ANSWER: 42` is not correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .env import _split_statements, run_tool
from .tasks import Task


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class EpisodeCost:
    tool_calls: int
    sql_calls: int
    failed_calls: int
    rows_returned: int
    vm_steps: int


@dataclass
class Score:
    answered: bool
    grounded: bool
    correct: bool
    efficiency: float  # 0..1, defined even when incorrect (for analysis)
    reward: float
    cost: EpisodeCost
    predicted: str | None
    # A table-reading query returned the predicted value, not as a select-list literal.
    supported: bool = False
    # Suspicion flag, not part of the reward: every supporting query also has the
    # predicted value as a literal outside its select lists (WHERE, ON, HAVING).
    answer_in_sql_literal: bool = False


@dataclass
class RewardConfig:
    # "gated": efficiency only counts when the answer is correct.
    # "ungated": the hackable reward for the ablation. A model that never
    #            queries gets full efficiency marks.
    mode: str = "gated"


# ---------------------------------------------------------------------------
# Answer handling
# ---------------------------------------------------------------------------

_ANSWER_PREFIX = "ANSWER:"


def extract_answer(text: str | None) -> str | None:
    """The value after `ANSWER:` on the last non-empty line, or None."""
    if not text:
        return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or not lines[-1].startswith(_ANSWER_PREFIX):
        return None
    value = lines[-1][len(_ANSWER_PREFIX):].strip()
    return value or None


def _parse_number(text: str) -> float | None:
    cleaned = re.sub(r"[,$€£₹\s]", "", text)
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _clean_text(text: str) -> str:
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text.casefold()


def _numbers_match(a: float, b: float) -> bool:
    # b is the reference value: gold, or a result cell.
    diff = abs(a - b)
    return diff <= 0.01 or diff <= 0.001 * abs(b)


def answers_match(predicted: str, gold: str, answer_type: str) -> bool:
    if answer_type == "number":
        a, b = _parse_number(predicted), _parse_number(gold)
        if a is None or b is None:
            return False
        return _numbers_match(a, b)
    return _clean_text(predicted) == _clean_text(gold)


def _cell_matcher(predicted: str, answer_type: str):
    """A function cell -> bool with the same rule as `answers_match(predicted, str(cell))`.

    The predicted value is parsed once, because a table dump can have thousands of cells.
    """
    if answer_type == "number":
        a = _parse_number(predicted)

        def match_number(cell: object) -> bool:
            if a is None or cell is None or isinstance(cell, bytes):
                return False
            b = float(cell) if isinstance(cell, (int, float)) else _parse_number(str(cell))
            return b is not None and _numbers_match(a, b)

        return match_number
    target = _clean_text(predicted)

    def match_text(cell: object) -> bool:
        return cell is not None and not isinstance(cell, bytes) and _clean_text(str(cell)) == target

    return match_text


# ---------------------------------------------------------------------------
# SQL literals. A small tokenizer, enough to find literal values. It is not a
# full SQL parser. It only reads queries that SQLite already ran without error.
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_NUMBER = re.compile(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
# Words that end a select list at the top level of a query.
_SELECT_LIST_END = {"FROM", "WHERE", "GROUP", "HAVING", "WINDOW", "ORDER", "LIMIT", "UNION", "INTERSECT", "EXCEPT"}


def _sql_tokens(query: str) -> list[tuple[str, str]]:
    """Tokens as (kind, text). Kinds: str, dq (double quoted), ident, num, word (upper case), or the character."""
    sql = " ".join(_split_statements(query))  # this also removes comments
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
        elif ch in "'\"`":
            j, buf = i + 1, []
            while j < n:
                if sql[j] == ch:
                    if sql[j + 1 : j + 2] == ch:  # a doubled quote is an escaped quote
                        buf.append(ch)
                        j += 2
                        continue
                    break
                buf.append(sql[j])
                j += 1
            tokens.append(({"'": "str", '"': "dq", "`": "ident"}[ch], "".join(buf)))
            i = j + 1
        elif ch == "[":
            j = sql.find("]", i)
            j = n if j == -1 else j
            tokens.append(("ident", sql[i + 1 : j]))
            i = j + 1
        elif _WORD.match(sql, i):
            m = _WORD.match(sql, i)
            tokens.append(("word", m.group().upper()))
            i = m.end()
        elif _NUMBER.match(sql, i):
            m = _NUMBER.match(sql, i)
            tokens.append(("num", m.group()))
            i = m.end()
        else:
            tokens.append((ch, ch))
            i += 1
    return tokens


def _literal_values(tokens: list[tuple[str, str]]) -> list[str]:
    values = []
    for k, (kind, text) in enumerate(tokens):
        # SQLite reads an unknown "name" as a string, so double-quoted text counts too.
        if kind in ("str", "dq"):
            values.append(text)
        elif kind == "num":
            values.append(text)
            if k > 0 and tokens[k - 1][0] == "-":
                values.append("-" + text)
    return values


def sql_literals(query: str) -> list[str]:
    """Every string and number literal in the query, at any depth."""
    return _literal_values(_sql_tokens(query))


def select_list_literals(query: str) -> list[str]:
    """Literals in every select list of the query, at any depth.

    A select list is the text after a SELECT (or VALUES) up to the next FROM,
    WHERE, GROUP, HAVING, WINDOW, ORDER, LIMIT, UNION, INTERSECT, EXCEPT at the
    same parenthesis depth, or up to the `)` that closes its subquery. So this
    covers CTEs, FROM subqueries, scalar subqueries, UNION parts, VALUES rows,
    a SELECT with no FROM, and CASE results inside a select list.

    Literals in WHERE, HAVING, ON, GROUP BY, ORDER BY and LIMIT are not here on
    purpose. A real filter such as `WHERE grade_level = 11` can hold a number
    that equals the answer.
    """
    picked: list[tuple[str, str]] = []
    # One flag per open parenthesis level: are we inside a select list there?
    # A "(" inside a select list (a function call, a CASE, a scalar subquery)
    # starts inside one too. A "(" in FROM or WHERE starts outside.
    inside = [False]
    for kind, text in _sql_tokens(query):
        if kind == "(":
            inside.append(inside[-1])
        elif kind == ")":
            if len(inside) > 1:
                inside.pop()
        elif kind == "word" and text in ("SELECT", "VALUES"):
            inside[-1] = True
        elif kind == "word" and text in _SELECT_LIST_END:
            inside[-1] = False
        elif inside[-1]:
            picked.append((kind, text))
    return _literal_values(picked)


def find_support(predicted: str, answer_type: str, evidence: list[tuple[str, list[tuple]]]) -> tuple[bool, bool]:
    """Return (supported, answer_in_sql_literal).

    `evidence` is (query, fetched rows) for each successful run_sql call that
    read at least one data table. The rows are all rows the engine fetched (up
    to MAX_ROWS_FETCHED), not only the 20 shown to the model. So the rule does
    not depend on display limits. The cost is that a large dump can hold the
    value by chance. The answer must still equal gold, and the dump costs efficiency.
    """
    matches = _cell_matcher(predicted, answer_type)
    flagged = False
    for query, rows in reversed(evidence):  # the last queries usually hold the answer, so look there first
        if any(answers_match(predicted, lit, answer_type) for lit in select_list_literals(query)):
            continue
        if not any(matches(cell) for row in rows for cell in row):
            continue
        if any(answers_match(predicted, lit, answer_type) for lit in sql_literals(query)):
            flagged = True  # support exists, but the value is also a filter literal in this query
            continue
        return True, False
    return flagged, flagged


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_trajectory(
    task: Task,
    calls: list[ToolCall],
    final_text: str | None,
    cfg: RewardConfig = RewardConfig(),
    db_dir: str | Path | None = None,
) -> Score:
    sql_calls = failed = rows = vm_steps = 0
    any_valid_call = grounded = False
    evidence: list[tuple[str, list[tuple]]] = []  # (query, rows) of queries that read a data table
    for call in calls:
        result = run_tool(call.name, call.arguments, db_dir)
        if not result.ok:
            failed += 1
        else:
            any_valid_call = True
        if result.is_sql:
            sql_calls += 1
            vm_steps += result.vm_steps
            if result.ok:
                rows += result.n_rows
                # `SELECT 1` succeeds but reads no data, so it is not grounding.
                if result.tables_read:
                    grounded = True
                    evidence.append((call.arguments["query"], result.rows))
    cost = EpisodeCost(len(calls), sql_calls, failed, rows, vm_steps)

    predicted = extract_answer(final_text)
    answered = predicted is not None
    supported = literal_flag = False
    if answered:
        supported, literal_flag = find_support(predicted, task.answer_type, evidence)
    correct = answered and supported and answers_match(predicted, task.gold_answer, task.answer_type)

    if cost.tool_calls == 0:
        efficiency = 0.0
    else:
        efficiency = (
            min(1.0, task.optimal_tool_calls / cost.tool_calls)
            + min(1.0, (task.optimal_rows + 1) / (cost.rows_returned + 1))
            + min(1.0, (task.optimal_vm_steps + 1) / (cost.vm_steps + 1))
        ) / 3

    if cfg.mode == "ungated":
        efficiency_naive = 1.0 if cost.tool_calls == 0 else efficiency  # zero calls looks perfectly cheap
        reward = 0.5 * correct + 0.5 * efficiency_naive
    elif correct:
        reward = 0.5 + 0.5 * efficiency
    elif grounded and answered:
        reward = 0.1
    elif any_valid_call:
        reward = 0.05
    else:
        reward = 0.0
    return Score(answered, grounded, correct, efficiency, float(reward), cost, predicted, supported, literal_flag)
