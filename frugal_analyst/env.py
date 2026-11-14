"""Tools for the agent: `get_schema` and `run_sql`, plus the VM-step meter.

The tool functions have type hints and Google-style docstrings on purpose.
AgentTune builds the JSON tool schema from them, and a small model reads them.
Tools never raise. Every failure comes back as text that starts with `ERROR:`.
"""

# No `from __future__ import annotations` here: the tools need real type
# objects (not strings) so AgentTune can build their JSON schema.

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .db import DOMAINS, ensure_db, schema_text

MAX_ROWS_SHOWN = 20
MAX_OUTPUT_CHARS = 1500
MAX_VM_STEPS = 5_000_000
MAX_ROWS_FETCHED = 2000
# SQLite calls our handler once per this many VM instructions. A small period
# keeps tiny queries from being counted as zero work.
PROGRESS_PERIOD = 10

ONLY_SELECT = "only SELECT queries are allowed"


@dataclass
class SqlResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    n_rows: int = 0
    vm_steps: int = 0
    error: str | None = None
    # Data tables the query read, from the SQLite authorizer. Sorted, no repeats.
    # The reward uses it: an answer must come from a query that read real data.
    tables_read: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def format_number(value: float) -> str:
    """Round to 2 decimals and drop trailing zeros: 42.0 -> '42', 3.14159 -> '3.14'."""
    text = f"{round(float(value), 2):.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def format_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return format_number(value)
    return str(value)


# ---------------------------------------------------------------------------
# Query checks
# ---------------------------------------------------------------------------


def _split_statements(query: str) -> list[str]:
    """Split on ';' that are outside quotes and comments. Empty parts are dropped."""
    parts, buf, i, n = [], [], 0, len(query)
    while i < n:
        ch = query[i]
        if ch in ("'", '"', "`"):  # quoted text or identifier
            end = query.find(ch, i + 1)
            while end != -1 and query[end + 1 : end + 2] == ch:  # doubled quote is an escape
                end = query.find(ch, end + 2)
            end = n - 1 if end == -1 else end
            buf.append(query[i : end + 1])
            i = end + 1
        elif ch == "[":
            end = query.find("]", i)
            end = n - 1 if end == -1 else end
            buf.append(query[i : end + 1])
            i = end + 1
        elif query.startswith("--", i):
            end = query.find("\n", i)
            end = n if end == -1 else end
            buf.append(" ")
            i = end
        elif query.startswith("/*", i):
            end = query.find("*/", i + 2)
            end = n if end == -1 else end + 2
            buf.append(" ")
            i = end
        elif ch == ";":
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(ch)
            i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _check_query(query: object) -> tuple[str | None, str | None]:
    """Return (statement, None) if the query is allowed, else (None, error)."""
    if not isinstance(query, str) or not query.strip():
        return None, "empty query"
    statements = _split_statements(query)
    if len(statements) != 1:
        return None, ONLY_SELECT if statements else "empty query"
    if not re.match(r"(select|with)\b", statements[0], re.IGNORECASE):
        return None, ONLY_SELECT
    return query, None


_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _authorizer(action, *_args) -> int:
    # Second line of defence behind the read-only file: no ATTACH, no PRAGMA, no writes.
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _is_data_table(name: object) -> bool:
    # sqlite_master and the other sqlite_* tables hold the schema, not data.
    return isinstance(name, str) and name != "" and not name.lower().startswith("sqlite_")


def _recording_authorizer(tables: set):
    """The same allow/deny rule as `_authorizer`, plus a record of each data table read.

    SQLite calls the authorizer while it compiles the query. For SQLITE_READ the
    first argument is the real table name. A CTE or a subquery is not a table,
    so a literal-only CTE reports nothing. `COUNT(*) FROM t` reports `t` once
    with an empty column name.
    """
    def check(action, arg1, *_rest) -> int:
        if action == sqlite3.SQLITE_READ and _is_data_table(arg1):
            tables.add(arg1)
        return _authorizer(action)

    return check


# ---------------------------------------------------------------------------
# Metered execution
# ---------------------------------------------------------------------------


def _fail(message: str, vm_steps: int = 0) -> SqlResult:
    return SqlResult(ok=False, error=message, vm_steps=vm_steps)


def execute_metered(database: str, query: str, db_dir: str | Path | None = None) -> SqlResult:
    """Run one read-only query and count SQLite VM steps. Never raises."""
    if database not in DOMAINS:
        return _fail(f"unknown database {database!r}. Valid databases: {', '.join(DOMAINS)}")
    statement, error = _check_query(query)
    if error:
        return _fail(error)
    steps = [0]
    tables: set[str] = set()

    def on_progress() -> int:
        steps[0] += 1
        return 1 if steps[0] * PROGRESS_PERIOD > MAX_VM_STEPS else 0  # non-zero aborts the query

    conn = None
    try:
        path = ensure_db(database, db_dir)
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        conn.set_authorizer(_recording_authorizer(tables))
        conn.set_progress_handler(on_progress, PROGRESS_PERIOD)
        cursor = conn.execute(statement)
        columns = [d[0] for d in cursor.description or []]
        rows = [tuple(r) for r in cursor.fetchmany(MAX_ROWS_FETCHED)]
        return SqlResult(ok=True, columns=columns, rows=rows, n_rows=len(rows), vm_steps=steps[0] * PROGRESS_PERIOD,
                         tables_read=tuple(sorted(tables)))
    except sqlite3.Error as exc:
        vm = steps[0] * PROGRESS_PERIOD
        text = str(exc)
        if "interrupted" in text:
            return _fail(f"query used too much engine work (limit {MAX_VM_STEPS} VM steps)", vm)
        if "not authorized" in text or "readonly" in text:
            return _fail(ONLY_SELECT, vm)
        return _fail(text, vm)
    except Exception as exc:  # the tool contract is: never raise
        return _fail(f"{type(exc).__name__}: {exc}", steps[0] * PROGRESS_PERIOD)
    finally:
        if conn is not None:
            conn.close()


def render_result(result: SqlResult) -> str:
    """Tool output text: header, up to 20 rows, a 'more rows' note, cut at 1500 chars."""
    if not result.ok:
        return f"ERROR: {result.error}"
    lines = [" | ".join(result.columns)]
    if result.n_rows == 0:
        lines.append("(no rows)")
    for row in result.rows[:MAX_ROWS_SHOWN]:
        lines.append(" | ".join(format_value(v) for v in row))
    if result.n_rows > MAX_ROWS_SHOWN:
        lines.append(f"... ({result.n_rows - MAX_ROWS_SHOWN} more rows not shown)")
    text = "\n".join(lines)
    if len(text) > MAX_OUTPUT_CHARS:
        marker = "\n... (output cut)"
        text = text[: MAX_OUTPUT_CHARS - len(marker)] + marker
    return text


# ---------------------------------------------------------------------------
# The two tools
# ---------------------------------------------------------------------------


def get_schema(database: str) -> str:
    """List the tables and columns of a database, with the row count of each table.

    Call this first. It does not show any data values.

    Args:
        database: Database name from the question, for example "retail".

    Returns:
        One line per table in the form table(column type, ...) and its row count.
    """
    return _schema_output(database)[0]


def run_sql(database: str, query: str) -> str:
    """Run one read-only SQL query (SQLite dialect) and return the result rows.

    Only a single SELECT or WITH statement is allowed. At most 20 rows are shown.
    Let SQL do the work: use COUNT, SUM, AVG, GROUP BY, ORDER BY and LIMIT
    instead of fetching many rows.

    Args:
        database: Database name from the question, for example "retail".
        query: One SQL SELECT statement.

    Returns:
        A header line with column names, then the rows. Errors start with "ERROR:".
    """
    return render_result(execute_metered(database, query))


TOOLS = [get_schema, run_sql]


def _schema_output(database: object, db_dir: str | Path | None = None) -> tuple[str, bool]:
    if database not in DOMAINS:
        return f"ERROR: unknown database {database!r}. Valid databases: {', '.join(DOMAINS)}", False
    try:
        return schema_text(database, db_dir), True
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}", False


# ---------------------------------------------------------------------------
# Tool dispatch used by the reward and the trace builder
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    output: str
    ok: bool
    n_rows: int = 0
    vm_steps: int = 0
    is_sql: bool = False
    # For run_sql only. `rows` holds every row the engine fetched (up to
    # MAX_ROWS_FETCHED), not only the 20 rows shown in `output`.
    rows: list[tuple] = field(default_factory=list)
    tables_read: tuple[str, ...] = ()


def run_tool(name: str, arguments: dict, db_dir: str | Path | None = None) -> ToolResult:
    """Execute one tool call by name against `db_dir`. Never raises."""
    try:
        if not isinstance(arguments, dict):
            return ToolResult("ERROR: arguments must be an object", False)
        # The live tools are plain functions, so an extra key makes the real
        # call fail. The reward must see the same failure, or a bad call would
        # count as a good one.
        allowed = {"get_schema": {"database"}, "run_sql": {"database", "query"}}
        extra = sorted(set(arguments) - allowed.get(name, set(arguments)))
        if extra:
            return ToolResult(f"ERROR: unexpected argument {extra[0]!r}", False, is_sql=(name == "run_sql"))
        if name == "get_schema":
            if "database" not in arguments:
                return ToolResult("ERROR: missing argument 'database'", False)
            text, ok = _schema_output(arguments["database"], db_dir)
            return ToolResult(text, ok)
        if name == "run_sql":
            for key in ("database", "query"):
                if key not in arguments:
                    return ToolResult(f"ERROR: missing argument {key!r}", False, is_sql=True)
            result = execute_metered(arguments["database"], arguments["query"], db_dir)
            return ToolResult(render_result(result), result.ok, result.n_rows, result.vm_steps, is_sql=True,
                              rows=result.rows, tables_read=result.tables_read)
        return ToolResult(f"ERROR: unknown tool {name!r}. Valid tools: get_schema, run_sql", False)
    except Exception as exc:
        return ToolResult(f"ERROR: {type(exc).__name__}: {exc}", False)
