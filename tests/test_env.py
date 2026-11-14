import inspect

import pytest

from frugal_analyst import env
from frugal_analyst.db import DOMAINS, ensure_dbs

ensure_dbs()


def test_tools_list_and_docstrings():
    assert env.TOOLS == [env.get_schema, env.run_sql]
    for tool in env.TOOLS:
        doc = inspect.getdoc(tool)
        assert doc and "Args:" in doc
        for param in inspect.signature(tool).parameters:
            assert f"{param}:" in doc
            assert inspect.signature(tool).parameters[param].annotation is str


def test_get_schema():
    text = env.get_schema("retail")
    assert text.startswith("customers(")
    assert "orders(" in text


def test_unknown_database():
    for out in (env.get_schema("nope"), env.run_sql("nope", "SELECT 1")):
        assert out.startswith("ERROR: unknown database")
        for name in DOMAINS:
            assert name in out


def test_run_sql_output_format():
    out = env.run_sql("retail", "SELECT code, description FROM status_codes ORDER BY code LIMIT 2")
    lines = out.splitlines()
    assert lines[0] == "code | description"
    assert len(lines) == 3
    assert lines[1].startswith("CXL_OOS | ")


def test_run_sql_limits_rows_shown():
    out = env.run_sql("retail", "SELECT order_id FROM orders")
    lines = out.splitlines()
    assert lines[0] == "order_id"
    # Only 2000 rows are fetched, and 20 of them are shown.
    assert lines[-1] == "... (1980 more rows not shown)"
    assert len([ln for ln in lines if ln.isdigit()]) == env.MAX_ROWS_SHOWN


def test_run_sql_cuts_long_output():
    out = env.run_sql("retail", "SELECT customer_id, name, city, segment, signup_date FROM customers, (SELECT 1 UNION SELECT 2) LIMIT 20")
    assert len(out) <= env.MAX_OUTPUT_CHARS
    wide = env.run_sql("retail", "SELECT " + ", ".join(["name || name || name || name"] * 6) + " FROM customers")
    assert len(wide) <= env.MAX_OUTPUT_CHARS
    assert wide.endswith("(output cut)")


def test_no_rows_and_null_rendering():
    assert env.run_sql("retail", "SELECT 1 WHERE 0").endswith("(no rows)")
    assert env.run_sql("retail", "SELECT NULL AS x") == "x\nNULL"
    assert env.run_sql("retail", "SELECT 3.14159, 2.0") .splitlines()[1] == "3.14 | 2"


@pytest.mark.parametrize("query", [
    "INSERT INTO status_codes VALUES ('X', 'x')",
    "DELETE FROM orders",
    "UPDATE orders SET quantity = 0",
    "DROP TABLE orders",
    "CREATE TABLE t(a)",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE '/tmp/x.sqlite' AS x",
    "SELECT 1; SELECT 2",
    "SELECT 1; DROP TABLE orders",
    "SELECT 1;\n-- comment\nDELETE FROM orders",
    "WITH x AS (SELECT 1) DELETE FROM orders",
    "WITH x AS (SELECT 1) INSERT INTO status_codes SELECT 'A', 'b'",
    "VACUUM",
    "  ",
])
def test_run_sql_refuses_unsafe_queries(query):
    out = env.run_sql("retail", query)
    assert out.startswith("ERROR:")
    if query.strip():
        assert out == "ERROR: only SELECT queries are allowed"


def test_semicolon_inside_string_is_fine():
    assert env.run_sql("retail", "SELECT 'a;b' AS x; ") == "x\na;b"


def test_refused_write_leaves_data_unchanged():
    before = env.run_sql("retail", "SELECT COUNT(*) FROM orders")
    env.run_sql("retail", "DELETE FROM orders")
    assert env.run_sql("retail", "SELECT COUNT(*) FROM orders") == before


@pytest.mark.parametrize("query", [None, 123, "", "SELEC", "SELECT * FROM no_such_table", "SELECT \x00", "SELECT '", b"SELECT 1", ["SELECT 1"], "SELECT " * 5000])
def test_run_sql_never_raises(query):
    out = env.run_sql("retail", query)
    assert isinstance(out, str)


def test_tools_never_raise_on_odd_database_values():
    assert env.get_schema(None).startswith("ERROR:")
    assert env.run_sql(["retail"], "SELECT 1").startswith("ERROR:")
    assert env.run_tool("run_sql", {"database": "retail"}).output.startswith("ERROR: missing argument")
    assert env.run_tool("nope", {}).ok is False
    assert env.run_tool("get_schema", "retail").ok is False


def test_vm_steps_are_deterministic():
    sql = "SELECT COUNT(*) FROM orders WHERE quantity >= 3"
    a, b = env.execute_metered("retail", sql), env.execute_metered("retail", sql)
    assert a.ok and a.vm_steps > 0
    assert a.vm_steps == b.vm_steps
    assert a.rows == b.rows


def test_select_star_costs_more_than_selective_count():
    big = env.execute_metered("retail", "SELECT * FROM orders")
    cheap = env.execute_metered("retail", "SELECT COUNT(*) FROM orders WHERE status = 'CXL_USR'")
    assert big.ok and cheap.ok
    assert big.n_rows == env.MAX_ROWS_FETCHED
    assert big.vm_steps > cheap.vm_steps


def test_expensive_query_is_aborted():
    result = env.execute_metered("retail", "SELECT COUNT(*) FROM orders a, orders b, orders c")
    assert not result.ok
    assert "too much engine work" in result.error
    assert result.vm_steps >= env.MAX_VM_STEPS
    assert env.run_sql("retail", "SELECT COUNT(*) FROM orders a, orders b, orders c").startswith("ERROR:")


def test_execute_metered_fields():
    r = env.execute_metered("retail", "SELECT order_id, quantity FROM orders LIMIT 5")
    assert (r.ok, r.columns, r.n_rows, r.error) == (True, ["order_id", "quantity"], 5, None)
    assert len(r.rows) == 5 and isinstance(r.rows[0], tuple)
    bad = env.execute_metered("retail", "SELECT nope FROM orders")
    assert bad.ok is False and bad.rows == [] and bad.n_rows == 0 and "nope" in bad.error


def test_with_queries_work():
    r = env.execute_metered("retail", "WITH t AS (SELECT quantity FROM orders) SELECT COUNT(*) FROM t")
    assert r.ok and r.rows[0][0] == 3000


def test_format_number():
    assert env.format_number(42.0) == "42"
    assert env.format_number(3.14159) == "3.14"
    assert env.format_number(-0.001) == "0"
    assert env.format_number(12.5) == "12.5"


def test_run_tool_rejects_unknown_arguments():
    from frugal_analyst.env import run_tool
    bad = run_tool("run_sql", {"database": "retail", "query": "SELECT 1", "limit": 5})
    assert not bad.ok and bad.output.startswith("ERROR:")
    bad = run_tool("get_schema", {"database": "retail", "verbose": True})
    assert not bad.ok and bad.output.startswith("ERROR:")
    assert run_tool("run_sql", {"database": "retail", "query": "SELECT 1"}).ok
