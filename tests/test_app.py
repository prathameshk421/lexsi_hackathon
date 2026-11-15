"""Tests for the demo app handlers. They call plain functions, so no browser is needed."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

import fa_backend  # noqa: E402
import fa_data  # noqa: E402
import fa_logic as logic  # noqa: E402
import fa_render as render  # noqa: E402
from frugal_analyst.db import DOMAINS, ensure_dbs  # noqa: E402
from frugal_analyst.env import execute_metered  # noqa: E402
from frugal_analyst.reward import ToolCall  # noqa: E402
from frugal_analyst.traces import write_traces  # noqa: E402

ensure_dbs()

HAS_TRACES = bool(fa_data.discover().get("test_id", {}).get("oracle")) and bool(fa_data.discover().get("test_id", {}).get("naive"))
needs_traces = pytest.mark.skipif(not HAS_TRACES, reason="scripted trace files are not generated")


def make_trace(task_id="retail-L1-0001", model="m", reward=1.0, rows=1, vm=100, calls=2, correct=True,
               question="How many orders?", sql="SELECT COUNT(*) FROM orders", output="n\n42", final="ANSWER: 42",
               domain="retail", level=1):
    steps = [{"type": "tool_call", "name": "get_schema", "arguments": {"database": domain}, "ok": True,
              "n_rows": 0, "vm_steps": 0, "output": "orders(...)"}]
    steps += [{"type": "tool_call", "name": "run_sql", "arguments": {"database": domain, "query": sql}, "ok": True,
               "n_rows": rows, "vm_steps": vm, "output": output}] * max(calls - 1, 0)
    steps.append({"type": "final", "text": final})
    return {
        "task_id": task_id, "domain": domain, "level": level, "split": "test_id", "question": question,
        "gold_answer": "42", "answer_type": "number", "model": model, "steps": steps, "predicted": "42",
        "answered": True, "grounded": True, "correct": correct, "efficiency": 0.5, "reward": reward,
        "cost": {"tool_calls": calls, "sql_calls": calls - 1, "failed_calls": 0, "rows_returned": rows, "vm_steps": vm},
        "optimal": {"tool_calls": 2, "rows": 1, "vm_steps": 100},
        "raw_completion": None, "completion_tokens": None, "latency_s": None,
    }


@pytest.fixture
def trace_dir(tmp_path):
    d = tmp_path / "traces"
    d.mkdir()
    write_traces(d / "zero-shot_test_id.jsonl", [
        make_trace("retail-L1-0001", reward=0.2, rows=500, vm=9000, calls=6, correct=False),
        make_trace("retail-L1-0002", reward=0.6, rows=20, vm=400, calls=3),
        make_trace("clinic-L2-0003", reward=0.9, rows=1, vm=110, calls=2, domain="clinic", level=2),
    ])
    write_traces(d / "grpo_test_id.jsonl", [
        make_trace("retail-L1-0001", reward=1.0),
        make_trace("retail-L1-0002", reward=0.95, rows=1, vm=100),
        make_trace("clinic-L2-0003", reward=0.9, rows=1, vm=110, domain="clinic", level=2),
    ])
    return d


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_discover_uses_only_fixed_names(tmp_path):
    d = tmp_path
    for name in ["grpo_test_id.jsonl", "Bad_test_id.jsonl", "a b_test_id.jsonl", "x_other.jsonl", "oracle_test_ood.jsonl",
                 "..hidden_test_id.jsonl", "zero-shot_train.jsonl"]:
        (d / name).write_text("")
    found = fa_data.discover(d)
    assert set(found) == {"test_id", "test_ood", "train"}
    assert set(found["test_id"]) == {"grpo"}
    assert fa_data.load_traces("test_id", "../../etc/passwd", d) == {}
    assert fa_data.load_traces("../x", "grpo", d) == {}
    assert fa_data.discover(d / "missing") == {}


def test_replay_assembles_two_sides(trace_dir):
    html = logic.replay_view("test_id", "zero-shot", "grpo", "retail-L1-0001", trace_dir)
    assert "zero-shot" in html and "grpo" in html
    assert "How many orders?" in html
    assert html.count("fa-card fa-side-") == 2
    assert "Cost at a glance" in html
    assert "Wrong" in html and "Correct" in html
    text = re.sub(r"<[^>]+>", "", html)  # SQL keywords are wrapped in colour spans
    assert "SELECT COUNT(*)\nFROM orders" in text  # SQL is shown with one clause per line
    assert '<span class="sq-k">SELECT</span>' in html
    assert "Both answers are correct." not in html and "Only <b>SFT + GRPO, 50 steps</b> is correct." in html  # the story
    assert "optimal 2" in html


def test_replay_empty_and_bad_inputs(tmp_path, trace_dir):
    assert "No traces yet" in logic.replay_view("test_id", "a", "b", "x", tmp_path / "none")
    assert "Pick two models" in logic.replay_view("test_id", "nope", "grpo", "retail-L1-0001", trace_dir)
    assert "No task selected" in logic.replay_view("test_id", "zero-shot", "grpo", "missing-id", trace_dir)
    assert "No task selected" in logic.replay_view("test_id", "zero-shot", "grpo", None, trace_dir)


def test_choices_filters_and_defaults(trace_dir):
    assert len(logic.task_choices("test_id", "zero-shot", "grpo", "all", "all", trace_dir)) == 3
    assert len(logic.task_choices("test_id", "zero-shot", "grpo", "clinic", "all", trace_dir)) == 1
    assert len(logic.task_choices("test_id", "zero-shot", "grpo", "all", "1", trace_dir)) == 2
    assert logic.task_choices("test_id", "zero-shot", "grpo", "../etc", "9", trace_dir) != []  # unknown filter means no filter
    st = logic.initial_compare_state(trace_dir)
    assert (st["a"], st["b"]) == ("zero-shot", "grpo")
    assert st["task"] == "retail-L1-0001"  # the largest difference
    assert [v for _, v in st["domains"]] == ["all", "retail", "clinic"]


def test_most_different_task(trace_dir):
    assert logic.most_different_task("test_id", "zero-shot", "grpo", trace_dir=trace_dir) == "retail-L1-0001"
    assert logic.most_different_task("test_id", "zero-shot", "grpo", "clinic", "all", trace_dir) == "clinic-L2-0003"
    assert logic.most_different_task("test_id", "zero-shot", "grpo", "school", "all", trace_dir) is None
    # reward ties fall back to rows
    a, b = make_trace("t1", reward=0.5, rows=10), make_trace("t1", reward=0.5, rows=900)
    assert logic.difference_score(fa_data.clean_trace(a), fa_data.clean_trace(b)) > (0.0, 0, 0, 0)


def test_random_task_and_filter_keeps_current(trace_dir):
    import random
    ids = {v for _, v in logic.task_choices("test_id", "zero-shot", "grpo", trace_dir=trace_dir)}
    assert logic.random_task("test_id", "zero-shot", "grpo", trace_dir=trace_dir, rng=random.Random(1)) in ids
    choices, pick = logic.on_filter_change("test_id", "zero-shot", "grpo", "all", "all", "retail-L1-0002", trace_dir)
    assert pick == "retail-L1-0002"
    _, pick = logic.on_filter_change("test_id", "zero-shot", "grpo", "clinic", "all", "retail-L1-0002", trace_dir)
    assert pick == "clinic-L2-0003"


def test_default_pair_rules():
    def agg(r):
        return {"tasks": 5, "mean_reward": r}
    assert fa_data.default_pair({"naive": agg(.6), "oracle": agg(1.0)}) == ("naive", "oracle")
    assert fa_data.default_pair({"zero-shot": agg(.1), "grpo": agg(.9), "oracle": agg(1.0), "naive": agg(.6)}) == ("zero-shot", "grpo")
    assert fa_data.default_pair({"zero-shot": agg(.1), "distilled": agg(.5), "oracle": agg(1.0)}) == ("zero-shot", "distilled")
    assert fa_data.default_pair({}) == (None, None)


@needs_traces
def test_replay_on_real_scripted_traces():
    st = logic.initial_compare_state()
    html = logic.replay_view(st["split"], st["a"], st["b"], st["task"])
    assert st["a"] in html and st["b"] in html and "fa-step" in html
    # Both sides must show the same task and the cost numbers from the trace fields.
    ta = fa_data.load_traces(st["split"], st["a"])[st["task"]]
    assert render.num(ta["cost"]["vm_steps"]) in html


def test_clean_trace_survives_garbage():
    assert fa_data.clean_trace(None) is None
    assert fa_data.clean_trace({"task_id": 5}) is None
    t = fa_data.clean_trace({"task_id": "x", "steps": "no", "cost": [], "level": "2", "reward": "high", "optimal": 3})
    assert t["steps"] == [] and t["level"] == 0 and t["reward"] == 0.0 and t["cost"]["tool_calls"] == 0
    t = fa_data.clean_trace({"task_id": "x", "steps": [{"type": "tool_call", "arguments": 5, "output": None}, 7, {"type": "?"}]})
    assert len(t["steps"]) == 1 and t["steps"][0]["arguments"] == {}
    assert render.replay_html(t, "a", t, "b")  # renders without error


def test_bad_lines_in_trace_file_are_skipped(tmp_path):
    (tmp_path / "grpo_test_id.jsonl").write_text(
        "not json\n" + json.dumps(make_trace("ok-1")) + "\n[]\n" + '{"task_id": 3}\n'
    )
    assert list(fa_data.load_traces("test_id", "grpo", tmp_path)) == ["ok-1"]


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------

EVIL = '<script>alert(1)</script><img src=x onerror=alert(2)> [link](javascript:alert(3)) **bold** `code` "q" \'s\''


def test_untrusted_text_is_rendered_inert(tmp_path):
    evil = make_trace("retail-L1-0001", model=EVIL, question=EVIL, sql="SELECT '" + EVIL + "'", output=EVIL, final=EVIL)
    evil["gold_answer"] = EVIL
    evil["steps"][0]["name"] = EVIL
    evil["steps"][0]["arguments"] = {"database": EVIL}
    evil["domain"] = EVIL
    evil["task_id"] = "retail-L1-0001"
    write_traces(tmp_path / "zero-shot_test_id.jsonl", [evil])
    write_traces(tmp_path / "grpo_test_id.jsonl", [evil])
    html = logic.replay_view("test_id", "zero-shot", "grpo", "retail-L1-0001", tmp_path)
    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert "onerror=alert(2)>" not in html  # the tag is escaped, so the attribute text stays as text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    # every "<" in the page starts one of our own tags
    own = set(re.findall(r"<\s*/?\s*([a-zA-Z0-9]+)", html))
    assert own <= {"div", "span", "section", "header", "pre", "code", "b", "h2", "h3", "p", "ol", "li", "details", "summary",
                   "dl", "dt", "dd"}
    # every attribute in the page is one we wrote: a fixed class, a width, a title, or aria-hidden
    for attrs in re.findall(r"<[a-z0-9]+(\s[^>]*)>", html):
        for name in re.findall(r'\s([a-zA-Z-]+)="', attrs):
            assert name in {"class", "style", "title", "aria-hidden", "role"}, name
    assert not re.search(r"<[a-z0-9]+[^>]*\son[a-z]+\s*=", html, re.I)
    # task dropdown labels are plain text, and the escape helper is exact
    assert render.esc("<b>&\"'") == "&lt;b&gt;&amp;&quot;&#x27;"


def test_evil_text_in_live_and_playground_views():
    t = fa_data.clean_trace(make_trace(question=EVIL, output=EVIL, final=EVIL))
    page = render.live_html(t, EVIL, 2, True, False, EVIL, EVIL)
    assert "<script" not in page.lower() and "<img" not in page.lower()
    assert "<script" not in render.query_stats_html(0, 0, False, "", EVIL).lower()
    assert "<script" not in render.notice_html(EVIL, "info", EVIL).lower()
    assert "<script" not in render.pair_html([(EVIL, EVIL, 1, 1, EVIL)]).lower()
    assert "<script" not in render.headline_html(EVIL, EVIL, [(EVIL, 1, 2, EVIL)], EVIL).lower()


SAFE_ATTRS = re.compile(r'^(\s+(class="[a-z0-9 -]*"|style="width:[0-9.]+%"|aria-hidden="true"|role="(status|alert|region)"))*$')


def tags_are_own(page: str, allowed: set) -> None:
    """Every tag is one of ours, and no tag carries an attribute that came from the text."""
    for name, attrs in re.findall(r"<\s*/?\s*([a-zA-Z0-9]+)([^>]*)>", page):
        assert name.lower() in allowed, name
        assert not re.search(r"\son[a-z]+\s*=", attrs, re.I), attrs


def test_sql_highlight_escapes_every_token():
    evil_sql = (
        "SELECT '</span><script>alert(1)</script>' AS \"<img src=x onerror=alert(2)>\" FROM t "
        "WHERE x = 1 -- <b onmouseover=alert(3)>\n</code></pre><script>alert(4)</script> <a href=javascript:alert(5)>x</a>"
    )
    out = render.highlight_sql(evil_sql)
    assert "<script" not in out.lower() and "<img" not in out.lower() and "<a " not in out.lower() and "<b " not in out
    assert "&lt;script&gt;" in out and "&lt;/code&gt;" in out
    for name, attrs in re.findall(r"<\s*/?\s*([a-zA-Z0-9]+)([^>]*)>", out):
        assert name == "span" and (attrs == "" or re.fullmatch(r' class="sq-[ksnfc]"', attrs)), (name, attrs)
    # the text is the same after the tags are taken out and the entities are read back
    import html as _html
    assert _html.unescape(re.sub(r"<[^>]+>", "", out)) == evil_sql
    # inside a step, with the long output behind a disclosure
    step = {"type": "tool_call", "name": "run_sql", "arguments": {"database": "retail", "query": evil_sql}, "ok": False,
            "n_rows": 0, "vm_steps": 0, "output": "\n".join([EVIL] * 30)}
    page = render.step_html(1, step)
    tags_are_own(page, {"li", "span", "div", "pre", "code", "details", "summary"})
    assert "Show full output (30 lines)" in page and "failed" in page


def test_new_views_keep_untrusted_text_inert():
    page = render.schema_html(f"t{EVIL}(a TEXT) -- 3 rows\n{EVIL}\nok_table(a INTEGER, b TEXT) -- 1,200 rows")
    tags_are_own(page, {"div", "details", "summary", "ul", "li", "span"})
    assert "ok_table" in page and "1,200 rows" in page and "&lt;script&gt;" in page
    table = render.table_html([EVIL, "n"], [[EVIL, 1.5], ["x", 20.25]], ["fa-row-scripted", EVIL])
    tags_are_own(table, {"div", "table", "thead", "tbody", "tr", "th", "td"})
    assert "&lt;script&gt;" in table and ">1.50<" in table and ">20.25<" in table and 'class="fa-num"' in table
    story = render.story_html(fa_data.clean_trace(make_trace(rows=500, vm=9000, calls=6)), EVIL,
                              fa_data.clean_trace(make_trace()), "oracle")
    tags_are_own(story, {"div", "p", "b", "span"})
    words = re.sub(r"<[^>]+>", "", story)
    assert "Both answers are correct." in words and "500x the rows" in words and "3.0x the tool calls" in words
    assert "90x the VM steps of Oracle (scripted)." in words  # every number comes from the two traces
    trace = fa_data.clean_trace(make_trace(question=EVIL, output=EVIL, final=EVIL))
    for page in (render.live_html(trace, EVIL, 1, False, False, EVIL, EVIL), render.trajectory_html(trace, EVIL)):
        tags_are_own(page, {"div", "span", "section", "header", "pre", "code", "b", "h2", "ol", "li", "details", "summary"})
    assert "<script" not in render.headline_html(EVIL, EVIL, [(EVIL, 1, 2, EVIL), ("tool calls", 5, 2)], EVIL).lower()
    assert "better" in render.headline_html("a", "b", [("mean tool calls", 5.0, 2.0)], "x")
    assert "worse" in render.headline_html("a", "b", [("accuracy", 0.9, 0.5)], "x")


def test_playground_and_results_tables_are_html(trace_dir, tmp_path):
    stats, table = logic.playground_view("retail", "SELECT name, unit_price FROM products LIMIT 3")
    assert "Rows returned" in stats and table.count("<tr") == 4 and 'class="fa-num"' in table
    stats, table = logic.playground_view("retail", "DROP TABLE orders")
    assert "Query refused or failed" in stats and "No rows to show" in table
    v = logic.results_view("all", trace_dir, tmp_path / "nope.json")
    html = logic.results_table_html(v)
    assert html.count("<tr") == 3 and "zero-shot" in html and "<th" in html
    assert "No results yet" in logic.results_table_html(logic.results_view("all", tmp_path / "x", tmp_path / "x.json"))
    assert "Schema: 4 tables" in logic.schema_view("retail") and "Pick a database" in logic.schema_view("../x")
    assert logic.split_options(["test_id", "weird"]) == [("Seen domains", "test_id"), ("weird", "weird")]
    first = logic.initial_ask_page(None)
    assert "Correct" in first and "fa-step" in first and "Running..." not in first


def test_footer_links_are_validated_and_hidden_when_unset():
    import app as app_module
    assert render.safe_url("https://github.com/org/repo") == "https://github.com/org/repo"
    for bad in ["javascript:alert(1)", "http://x.org", 'https://x.org/"onmouseover=alert(1)', "https://x.org/<script>",
                "https://x .org", None, 5, "", "data:text/html,hi"]:
        assert render.safe_url(bad) is None, bad
    page = render.footer_html(app_module.EVENT, EVIL, [("GitHub repository", "https://github.com/org/repo"),
                                                      (EVIL, "javascript:alert(1)"), ("Write-up", "")])
    assert page.count("<a ") == 1 and 'href="https://github.com/org/repo"' in page and "noopener" in page
    assert "<script" not in page.lower() and "Agentic Fine-tuning track" in page
    empty = render.footer_html(app_module.EVENT, "m", app_module.footer_links({}))
    assert "<a " not in empty and "<nav" not in empty
    links = app_module.footer_links({"FRUGAL_LINK_WRITEUP": "https://example.org/post"})
    assert ("Write-up", "https://example.org/post") in links
    assert app_module.base_model_name({}) == fa_backend.DEFAULT_MODEL_ID


def test_format_sql_keeps_meaning_and_quotes():
    sql = "SELECT a, COUNT(*)   FROM t JOIN u ON t.id=u.id WHERE x = 'a  FROM  b' GROUP BY a ORDER BY 2 DESC LIMIT 1"
    out = render.format_sql(sql)
    assert out.count("\n") >= 4
    assert "'a  FROM  b'" in out  # text inside quotes is untouched
    assert re.sub(r"\s+", " ", out) == re.sub(r"\s+", " ", sql)


# ---------------------------------------------------------------------------
# SQL playground: refusals and limits
# ---------------------------------------------------------------------------

REFUSED = [
    "INSERT INTO orders (order_id) VALUES (999999)",
    "UPDATE orders SET quantity = 0",
    "DELETE FROM orders",
    "DROP TABLE orders",
    "CREATE TABLE evil (x INTEGER)",
    "ALTER TABLE orders ADD COLUMN x INTEGER",
    "PRAGMA table_info(orders)",
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE ':memory:' AS other",
    "ATTACH DATABASE '/tmp/evil.db' AS other",
    "VACUUM",
    "SELECT 1; DROP TABLE orders",
    "SELECT 1; SELECT 2",
    "SELECT 1; DELETE FROM orders",
    "WITH x AS (SELECT 1) DELETE FROM orders",
    "WITH x AS (SELECT 1) INSERT INTO orders (order_id) SELECT 1",
    "SELECT 1; PRAGMA table_info(orders)",
    "REPLACE INTO orders (order_id) VALUES (1)",
]


@pytest.mark.parametrize("sql", REFUSED)
def test_playground_refuses_writes_pragma_attach_and_stacked(sql):
    before = execute_metered("retail", "SELECT COUNT(*) FROM orders").rows
    stats, table = logic.playground_run("retail", sql)
    assert "Query refused or failed" in stats
    assert len(table) == 0
    assert execute_metered("retail", "SELECT COUNT(*) FROM orders").rows == before
    assert execute_metered("retail", "SELECT name FROM sqlite_master WHERE name = 'evil'").rows == []


def test_playground_runs_select_and_hints_on_many_rows():
    stats, table = logic.playground_run("retail", "SELECT COUNT(*) FROM orders")
    assert "Rows returned" in stats and "VM steps" in stats and "fa-hint" not in stats
    assert table.shape == (1, 1)
    stats, table = logic.playground_run("retail", "SELECT * FROM orders")
    assert "fa-hint" in stats and "rows of context" in stats
    assert len(table) == logic.MAX_TABLE_ROWS  # the table is capped
    assert "first 200 rows" in stats


def test_playground_vm_step_limit_is_enforced():
    heavy = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100000000) SELECT COUNT(*) FROM c")
    stats, table = logic.playground_run("retail", heavy)
    assert "too much engine work" in stats and len(table) == 0


def test_playground_validates_database_and_length():
    stats, _ = logic.playground_run("../../etc/passwd", "SELECT 1")
    assert "Pick a database" in stats
    stats, _ = logic.playground_run(None, "SELECT 1")
    assert "Pick a database" in stats
    stats, _ = logic.playground_run("retail", "SELECT 1 " + " " * logic.MAX_SQL_CHARS)
    assert "too long" in stats
    stats, _ = logic.playground_run("retail", "   ")
    assert "Type a SELECT" in stats
    stats, _ = logic.playground_run("retail", None)
    assert "Type a SELECT" in stats
    for d in DOMAINS:
        assert logic.valid_database(d) == d
    assert logic.valid_database("retail; DROP") is None


def test_example_pairs_same_answer_cheaper_cost():
    for i, p in enumerate(logic.EXAMPLE_PAIRS):
        cheap = execute_metered(p["database"], p["cheap"])
        costly = execute_metered(p["database"], p["costly"])
        assert cheap.ok and costly.ok, p["name"]
        same = costly.n_rows if p.get("same") == "row_count" else costly.rows[0][0]
        assert cheap.rows[0][0] == same, p["name"]
        assert cheap.n_rows < costly.n_rows or cheap.vm_steps < costly.vm_steps, p["name"]
        assert cheap.vm_steps < costly.vm_steps
        html = logic.run_pair(str(i))
        assert "Cheap" in html and "Costly" in html and html.count("<section") == 2
    assert logic.load_pair("0") == ("retail", logic.EXAMPLE_PAIRS[0]["cheap"])
    assert "Pick an example pair" in logic.run_pair("nope")
    assert "Pick an example pair" in logic.run_pair("99")


def test_cost_hint():
    assert logic.cost_hint(1, False) == ""
    assert "2000+" in logic.cost_hint(2000, True)


# ---------------------------------------------------------------------------
# Ask (live)
# ---------------------------------------------------------------------------


class FakeBackend:
    name = "fake-model"

    def __init__(self, calls=None, final="ANSWER: 1", fail=False):
        self.calls, self.final, self.fail, self.ran = calls, final, fail, 0

    def available(self):
        return True

    def run(self, database, question):
        self.ran += 1
        if self.fail:
            raise RuntimeError("boom with <b>secret path /etc</b>")
        return self.calls or [{"name": "get_schema", "arguments": {"database": database}},
                              {"name": "run_sql", "arguments": {"database": database, "query": "SELECT COUNT(*) FROM orders"}}], self.final, "raw"


def last(gen):
    out = None
    for out in gen:
        pass
    return out


def a_retail_example():
    ex = logic.example_choices("retail")
    assert ex
    return ex[0][1], logic.question_for_example("retail", ex[0][1])


def test_ask_oracle_and_naive_on_example_question():
    _, q = a_retail_example()
    pages = list(logic.ask_run("retail", q, "oracle", None, pace=False))
    assert len(pages) >= 3  # the steps arrive one after the other
    assert "Running..." in pages[0] and "Running..." not in pages[-1]
    assert pages[-1].count("fa-step") > pages[0].count("fa-step")
    assert "Correct" in pages[-1] and "optimal" in pages[-1]
    naive = last(logic.ask_run("retail", q, "naive", None, pace=False))
    oracle_trace = last(logic.ask_run("retail", q, "oracle", None, pace=False))
    assert naive != oracle_trace and "Correct" in naive


def test_ask_typed_question_with_scripted_policy_is_explained_not_failed():
    page = last(logic.ask_run("retail", "How many bananas are there?", "oracle", None, pace=False))
    assert "needs the gold path" in page and "Typed question" in page
    assert "live model, which is off" in page


def test_ask_validation_messages():
    assert "Pick a database" in last(logic.ask_run("nope", "q", "oracle", None, pace=False))
    assert "Pick an example question" in last(logic.ask_run("retail", "  ", "oracle", None, pace=False))
    assert "too long" in last(logic.ask_run("retail", "x" * (logic.MAX_QUESTION_CHARS + 1), "oracle", None, pace=False))
    assert "Pick a policy" in last(logic.ask_run("retail", "q", "bogus", None, pace=False))
    assert "Live model inference is off" in last(logic.ask_run("retail", "q", "model", None, pace=False))
    assert "Live model inference is off" in last(logic.ask_run("retail", "q", "model", fa_backend.NoBackend(), pace=False))


def test_ask_model_typed_question_shows_cost_without_verdict():
    backend = FakeBackend()
    page = last(logic.ask_run("retail", "How many orders are there?", "model", backend, pace=False))
    assert backend.ran == 1
    assert "No gold answer, so no verdict" in page
    assert "Correct" not in page and "Efficiency" not in page and "Reward" not in page
    assert "Rows returned" in page and "VM steps" in page and "fake-model" in page


def test_ask_model_on_example_question_is_scored():
    _, q = a_retail_example()
    page = last(logic.ask_run("retail", q, "model", FakeBackend(), pace=False))
    assert "Efficiency" in page and ("Correct" in page or "Wrong" in page)


def test_ask_model_errors_are_friendly():
    page = last(logic.ask_run("retail", "How many orders?", "model", FakeBackend(fail=True), pace=False))
    assert "could not finish" in page
    assert "boom" not in page and "/etc" not in page and "Traceback" not in page


def test_ask_model_calls_are_normalised_and_capped():
    many = [ToolCall("run_sql", {"database": "retail", "query": "SELECT 1"})] * 30 + ["junk", 5]
    assert len(logic.normalize_calls(many)) == logic.MAX_MODEL_CALLS
    # a model that asks for a write is refused by the env, and shows as a failed step
    evil = [{"name": "run_sql", "arguments": {"database": "retail", "query": "DROP TABLE orders"}}]
    page = last(logic.ask_run("retail", "typed question", "model", FakeBackend(calls=evil), pace=False))
    assert "failed" in page and "only SELECT queries are allowed" in page
    assert execute_metered("retail", "SELECT COUNT(*) FROM orders").ok


def test_policy_choices_and_backend_note():
    assert [v for _, v in logic.policy_choices(None)] == ["oracle", "naive"]
    assert [v for _, v in logic.policy_choices(fa_backend.NoBackend())] == ["oracle", "naive"]
    assert [v for _, v in logic.policy_choices(FakeBackend())] == ["oracle", "naive", "model"]
    assert "Live model inference is off" in logic.backend_note(None) and "Compare tab" in logic.backend_note(None)
    assert "is on" in logic.backend_note(FakeBackend())


def test_example_lookup_and_schema():
    assert logic.example_choices("nope") == []
    assert logic.example_task("retail", "no-such-id") is None
    assert logic.schema_for("nope") == "Pick a database."
    assert "orders(" in logic.schema_for("retail")
    for d in DOMAINS:
        assert logic.example_choices(d), d


# ---------------------------------------------------------------------------
# Missing runner and backend
# ---------------------------------------------------------------------------


def test_backend_off_without_runner_or_flag(monkeypatch):
    b = fa_backend.RunnerBackend()
    monkeypatch.delenv("FRUGAL_LIVE_MODEL", raising=False)
    assert b.available() is False  # no opt in
    monkeypatch.setenv("FRUGAL_LIVE_MODEL", "1")
    monkeypatch.setitem(sys.modules, "frugal_analyst.runner", None)  # makes `import` raise ImportError
    assert fa_backend.RunnerBackend().available() is False
    monkeypatch.undo()
    assert fa_backend.NoBackend().available() is False
    with pytest.raises(RuntimeError):
        fa_backend.NoBackend().run("retail", "q")


def test_backend_off_when_torch_is_missing(monkeypatch):
    monkeypatch.setenv("FRUGAL_LIVE_MODEL", "1")
    monkeypatch.setattr(fa_backend, "REQUIRED_MODULES", ("definitely_not_installed_module_xyz",))
    assert fa_backend.RunnerBackend().available() is False


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_results_missing_file_falls_back_to_traces(trace_dir, tmp_path):
    v = logic.results_view("all", trace_dir, tmp_path / "nope.json")
    assert v["source"] == "traces"
    assert "No evaluation results yet" in v["note"] and "computed from the trace files" in v["note"]
    by_model = {r["raw"]: r for r in v["rows"]}
    assert set(by_model) == {"zero-shot", "grpo"}
    assert by_model["zero-shot"]["n"] == 3
    assert by_model["zero-shot"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)  # computed from the trace fields
    assert by_model["grpo"]["tool_calls"] == 2
    assert "computed from traces" in v["headline"] and "Zero-shot (base model) to SFT + GRPO, 50 steps" in v["headline"]
    assert len(logic.results_table(v)) == 2
    chart = logic.results_chart(v, "Task success")
    assert list(chart["name"]) == ["Zero-shot (base model)", "SFT + GRPO, 50 steps"]


def test_results_nothing_at_all(tmp_path):
    v = logic.results_view("all", tmp_path / "none", tmp_path / "none.json")
    assert v["source"] == "none" and "No results yet" in v["note"]
    assert len(logic.results_table(v)) == 0
    assert len(logic.results_chart(v, "Accuracy")) == 0


@pytest.mark.parametrize("content", ["not json", "[]", "{}", '"text"', "null", '{"splits": 5}', '{"splits": {"a": {"models": 3}}}',
                                      "[" * 5000])
def test_results_unexpected_shapes_do_not_crash(content, trace_dir, tmp_path):
    f = tmp_path / "results.json"
    f.write_text(content)
    v = logic.results_view("all", trace_dir, f)
    assert v["source"] in ("traces", "none")
    logic.results_table(v)


def test_results_file_in_the_eval_agent_shape(trace_dir, tmp_path):
    def entry(acc, calls):
        return {"n": 10, "metrics": {"accuracy": acc, "reward": acc * 0.9, "tool_calls": calls, "weird": None, "x": "<b>"}}
    doc = {
        "metric_info": {"accuracy": {"label": "accuracy"}, "tool_calls": {"label": "mean tool calls"}},
        "splits": {"test_id": {"models": {"zero-shot": entry(0.4, 6.0), "grpo": entry(0.8, 2.5)}},
                   "test_ood": {"models": {"zero-shot": entry(0.3, 6.5), "grpo": entry(0.7, 2.9)}}},
        "comparisons": {"test_id": {"baseline": "zero-shot", "pairs": [{"model": "grpo", "metrics": {
            "accuracy": {"baseline_mean": 0.4, "candidate_mean": 0.8, "delta": 0.4, "ci_low": 0.3, "ci_high": 0.5, "verdict": "better"}}}]}},
    }
    f = tmp_path / "results.json"
    f.write_text(json.dumps(doc))
    v = logic.results_view("all", trace_dir, f)
    assert v["source"] == "results" and len(v["rows"]) == 4
    assert "Seen domains: Zero-shot (base model) to SFT + GRPO, 50 steps" in v["headline"] and "better" in v["headline"]
    assert "+0.300 to +0.500" in v["headline"]  # the interval comes from the file
    assert "Tool calls" in v["metrics"]
    one = logic.results_view("test_ood", trace_dir, f)
    assert len(one["rows"]) == 2 and "Test set: Held-out domain" in one["note"]
    assert len(logic.results_chart(v, "Task success")) == 4
    # the generic reader handles other shapes too
    g = tmp_path / "other.json"
    g.write_text(json.dumps({"a": {"zero-shot": {"accuracy": 0.1, "tool_calls": 5}, "grpo": {"accuracy": 0.9, "tool_calls": 2}}}))
    v = logic.results_view("all", trace_dir, g)
    assert v["source"] == "results" and len(v["rows"]) == 2


def test_real_results_file_if_present():
    v = logic.results_view("all")
    assert v["source"] in ("results", "traces", "none")
    logic.results_table(v)


@pytest.fixture
def five_label_dir(tmp_path):
    d = tmp_path / "traces"
    d.mkdir()
    spec = {"zero-shot": (0.2, False), "distilled": (0.6, True), "grpo": (0.95, True), "naive": (0.6, True), "oracle": (1.0, True)}
    for label, (reward, ok) in spec.items():
        for split in ("test_id", "test_ood"):
            tr = [make_trace(f"retail-L1-000{i}", model=label, reward=reward, correct=ok) for i in range(1, 4)]
            if label == "zero-shot":
                tr[0]["latency_s"] = 1.5
            write_traces(d / f"{label}_{split}.jsonl", tr)
    return d


def test_default_pair_is_zero_shot_vs_grpo_and_models_come_before_scripted(five_label_dir):
    st = logic.initial_compare_state(five_label_dir)
    assert (st["a"], st["b"]) == ("zero-shot", "grpo")
    # dropdown order: baseline, SFT, GRPO, then scripted reference policies, with plain names and the label
    assert st["labels"] == [("Zero-shot (base model) [zero-shot]", "zero-shot"), ("SFT, 274 tasks [distilled]", "distilled"),
                            ("SFT + GRPO, 50 steps [grpo]", "grpo"), ("Oracle (scripted) [oracle]", "oracle"),
                            ("Naive (scripted) [naive]", "naive")]
    v = logic.results_view("test_id", five_label_dir, five_label_dir / "none.json")
    assert [r["raw"] for r in v["rows"]] == ["zero-shot", "distilled", "grpo", "oracle", "naive"]
    assert [r["model"] for r in v["rows"]][-2:] == ["Oracle (scripted)", "Naive (scripted)"]
    assert [r["kind"] for r in v["rows"]] == ["model"] * 3 + ["scripted reference"] * 2
    assert "scripted reference policies, not models" in v["note"]
    assert "Zero-shot (base model) to SFT + GRPO, 50 steps" in v["headline"]
    chart = logic.results_chart(v, "Efficiency, correct answers only")
    assert list(chart["name"])[-1] == "Naive (scripted)" and list(chart["group"]).count("scripted reference") == 2
    html = logic.replay_view("test_id", "naive", "grpo", "retail-L1-0001", five_label_dir)
    assert "scripted reference" in html
    all_splits = logic.results_view("all", five_label_dir, five_label_dir / "none.json")
    assert all_splits["rows"][0]["name"] == "Zero-shot (base model)"  # traces aggregate over splits for 'all'


def test_results_short_table_and_latency_validity(five_label_dir, tmp_path):
    v = logic.results_view("test_id", five_label_dir, five_label_dir / "none.json")
    short = [h for _, h in v["headers"]]
    assert short[:4] == ["Model", "Label", "Tasks", "Task success"]
    assert "Tool-call validity" in short and "Latency (s)" in short  # latency exists for zero-shot only
    df = logic.results_table(v)
    assert df.loc[df["Label"] == "zero-shot", "Latency (s)"].iloc[0] == 1.5
    assert df.loc[df["Label"] == "zero-shot", "Tool-call validity"].iloc[0] == 1.0
    assert all(w >= 96 for w in logic.table_widths(df))
    assert logic.default_metric(v) == "Task success"
    # latency is hidden when no trace has it
    v2 = logic.results_view("all", tmp_path / "none", tmp_path / "none.json")
    assert v2["source"] == "none"


def test_results_json_short_and_full_tables(trace_dir, tmp_path):
    def entry(acc):
        m = {"accuracy": acc, "reward": acc, "efficiency": 0.5, "tool_calls": 3.0, "rows_returned": 10.0, "vm_steps": 900.0,
             "tool_call_validity": 0.9, "latency_s": 2.25, "completion_tokens": None, "failed_calls": 0.3, "call_ratio": 1.5,
             "grounded_rate": 1.0, "answered_rate": 1.0}
        return {"n": 5, "metrics": m}
    doc = {"splits": {"test_id": {"models": {"grpo": entry(0.9), "zero-shot": entry(0.4), "oracle": entry(1.0)}}}}
    f = tmp_path / "results.json"
    f.write_text(json.dumps(doc))
    v = logic.results_view("all", trace_dir, f)
    short = [h for _, h in v["headers"]]
    # honest measures first; with no correct-only efficiency in the file, the raw one is shown and named as such
    assert short == ["Model", "Label", "Split", "Tasks", "Task success", "Tool calls", "Latency (s)", "Tool-call validity",
                     "Reward", "Rows", "VM steps", "Efficiency, all runs (wrong ones too)"]
    full = [h for _, h in v["all_headers"]]
    assert len(full) > len(short) and "Failed calls" in full and "Grounded rate" in full
    assert [r["raw"] for r in v["rows"]] == ["zero-shot", "grpo", "oracle"]
    assert logic.results_table(v).shape == (3, len(short))
    assert logic.results_table(v, full=True).shape == (3, len(full))
    assert "Completion tokens" not in full  # null values are not shown


def test_chart_axis_starts_at_zero_and_has_headroom():
    import pandas as pd
    ones = pd.DataFrame({"name": ["a", "b"], "value": [1.0, 1.0], "group": ["x", "x"]})
    assert logic.axis_upper(ones) == 1.0
    counts = pd.DataFrame({"name": ["a", "b"], "value": [2.0, 4.6], "group": ["x", "x"]})
    assert logic.axis_upper(counts) > 4.6
    assert logic.axis_upper(pd.DataFrame({"name": [], "value": [], "group": []})) >= 1.0
    # the HTML chart: equal values still draw full, visible bars, and the axis labels start at 0
    view = {"rows": [{"name": "a", "raw": "grpo", "kind": "model", "split": "test_id", "accuracy": 1.0},
                     {"name": "b", "raw": "naive", "kind": "scripted reference", "split": "test_ood", "accuracy": 1.0}],
            "metric_keys": {"Accuracy": "accuracy"}}
    html = logic.chart_html(view, "Accuracy")
    assert html.count("width:100.0%") == 2 and "fa-hatch" in html and "<span>0</span>" in html and "<span>1</span>" in html
    assert "Seen domains" in html and "Held-out domain" in html  # split is the colour legend
    assert "No chart yet" in logic.chart_html({"rows": [], "metric_keys": {}}, "x")
    assert "<script" not in render.chart_html("<script>", [("<script>", 1.0, "<script>", False)], 1.0).lower()


def test_cost_colours_neutral_at_optimal_warm_above():
    cheap = fa_data.clean_trace(make_trace(rows=1, vm=100, calls=2))
    costly = fa_data.clean_trace(make_trace(rows=500, vm=9000, calls=6))
    html = render.meters_html(costly, "naive", cheap, "oracle")
    assert "fa-bar-warn" in html and "fa-bar-ok" in html
    assert html.count("fa-bar-warn") == 3 and html.count("fa-bar-ok") == 3  # naive over on all three, oracle at optimal on all three
    assert "x optimal" in html and "at optimal" in html  # the numbers keep carrying the meaning
    pair = render.pair_html([("Cheap", "SELECT 1", 1, 10, None), ("Costly", "SELECT 2", 300, 9000, None)])
    assert pair.count("fa-bar-warn") == 2 and pair.count("fa-bar-ok") == 2
    assert "fa-pair-cheap" in pair and "fa-pair-costly" in pair
    assert "fa-bar-a" not in html + pair and "fa-bar-b" not in html + pair


def test_output_blocks_are_plain():
    css = render.CSS
    assert "opacity:.9" not in css and "animation:none" in css
    assert ".fa-out{max-height:170px;overflow:auto}" in css


# ---------------------------------------------------------------------------
# App wiring and source rules
# ---------------------------------------------------------------------------

APP_FILES = sorted((ROOT / "app").glob("*.py"))


def test_app_builds_without_a_model():
    import app as app_module

    demo = app_module.build_demo(fa_backend.NoBackend())
    assert demo is not None
    assert "Frugal Analyst" in app_module.HERO


def test_app_source_rules():
    banned = [r"(?<![.\w])eval\(", r"(?<![.\w])exec\(", r"\bpickle\b", r"\bsubprocess\b", r"os\.system", r"share\s*=\s*True",
              r"allowed_paths", r"\bsqlite3\b", r"hf_[A-Za-z0-9]{20,}", r"\bmcp_server\s*=\s*True"]
    for path in APP_FILES:
        text = path.read_text(encoding="utf-8")
        for pat in banned:
            assert not re.search(pat, text), f"{path.name} matches {pat}"
        assert "—" not in text and "–" not in text, f"{path.name} has a dash character used as punctuation"
    for name in ("README.md", "requirements.txt"):
        assert "—" not in (ROOT / "app" / name).read_text(encoding="utf-8")


def test_only_env_touches_the_database():
    # No app module opens a database. All SQL goes through frugal_analyst.env.
    for path in APP_FILES:
        text = path.read_text(encoding="utf-8")
        assert "sqlite3" not in text and ".connect(" not in text


def test_launch_settings_are_conservative():
    text = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
    assert "share=False" in text and "show_error=False" in text
    assert "demo.queue(" in text and "default_concurrency_limit" in text and "max_size" in text


# ---------------------------------------------------------------------------
# Default pair, plain names, wrong-answer efficiency, honest chart default
# ---------------------------------------------------------------------------


def _agg(reward):
    return {"tasks": 5, "mean_reward": reward}


def test_default_pair_prefers_baseline_against_final_then_sft():
    every = {"zero-shot-stop": _agg(0.02), "zero-shot": _agg(0.1), "distilled": _agg(0.8), "grpo": _agg(0.85),
             "distilled-all": _agg(0.99), "grpo-mixed": _agg(0.8), "oracle": _agg(1.0), "naive": _agg(0.6)}
    assert fa_data.default_pair(every) == ("zero-shot", "grpo")
    # The bug seen in a browser: without grpo, the old rule picked the worst and best run by reward.
    no_grpo = {k: v for k, v in every.items() if k != "grpo"}
    assert fa_data.default_pair(no_grpo) == ("zero-shot", "distilled")
    # No baseline: worst and best main model, experiments left out while two main models exist.
    assert fa_data.default_pair({"distilled": _agg(0.8), "grpo": _agg(0.85), "distilled-all": _agg(0.99)}) == ("distilled", "grpo")
    assert fa_data.default_pair({"zero-shot-stop": _agg(0.02), "distilled-all": _agg(0.99), "oracle": _agg(1.0)}) == (
        "zero-shot-stop", "distilled-all")
    assert fa_data.default_pair({"new-run": _agg(0.3), "other-run": _agg(0.4)}) == ("new-run", "other-run")


def test_display_names_come_from_one_table_with_fallback():
    assert fa_data.display_label("grpo") == "SFT + GRPO, 50 steps"
    assert fa_data.display_label("zero-shot") == "Zero-shot (base model)"
    assert fa_data.display_label("distilled-all") == "SFT, all 400 tasks (reference)"
    assert fa_data.display_label("oracle") == "Oracle (scripted)"
    assert fa_data.display_label("my-new-run") == "my-new-run"  # unknown labels stay as they are
    assert fa_data.choice_label("grpo") == "SFT + GRPO, 50 steps [grpo]" and fa_data.choice_label("x-1") == "x-1"
    for label in ("zero-shot", "distilled", "grpo", "grpo-ungated", "distilled-mixed", "grpo-mixed", "zero-shot-stop",
                  "distilled-all", "oracle", "naive"):
        assert fa_data.run_note(label), label
        assert "—" not in fa_data.run_note(label) + fa_data.display_label(label)
    order = sorted(["naive", "zero-shot-stop", "grpo", "distilled-all", "zero-shot", "unknown", "distilled"],
                   key=fa_data.label_sort_key)
    assert order == ["zero-shot", "distilled", "grpo", "zero-shot-stop", "distilled-all", "unknown", "naive"]
    legend = render.runs_legend_html([(fa_data.display_label(l), l, fa_data.run_note(l)) for l in ("grpo", EVIL)])
    tags_are_own(legend, {"div", "details", "summary", "dl", "dt", "dd", "span"})
    assert "What are these runs" in legend and "50 GRPO steps" in legend


def test_wrong_answer_shows_efficiency_not_applicable(trace_dir):
    # zero-shot is wrong on retail-L1-0001 and its trace says efficiency 0.5
    html = logic.replay_view("test_id", "zero-shot", "grpo", "retail-L1-0001", trace_dir)
    assert html.count('class="fa-cell fa-cell-na"') == 1 and "counts only for a correct answer" in html
    right = fa_data.clean_trace(make_trace(correct=True))
    wrong = fa_data.clean_trace(make_trace(correct=False))
    assert "n/a" not in render.summary_html(right) and "0.50" in render.summary_html(right)
    assert "n/a" in render.summary_html(wrong) and "0.50" not in render.summary_html(wrong).split("Reward")[0]
    assert "Efficiency" not in render.summary_html(wrong, has_gold=False)  # no gold, no efficiency at all
    # A cheap wrong run: the meters do not show it as a neutral "at optimal" bar, and the story says it earns nothing.
    cheap_wrong = fa_data.clean_trace(make_trace(rows=1, vm=100, calls=2, correct=False))
    costly_right = fa_data.clean_trace(make_trace(rows=500, vm=9000, calls=6, correct=True))
    meters = render.meters_html(cheap_wrong, "zero-shot-stop", costly_right, "grpo")
    assert "fa-bar-na" in meters and "no correct answer" in meters and "cost does not count" in meters
    story = re.sub(r"<[^>]+>", "", render.story_html(costly_right, "grpo", cheap_wrong, "zero-shot-stop"))
    assert "3.0x the tool calls" in story and "gave no correct answer, so the lower cost earns nothing" in story
    both_wrong = re.sub(r"<[^>]+>", "", render.story_html(cheap_wrong, "a", fa_data.clean_trace(
        make_trace(rows=9, correct=False)), "b"))
    assert "Neither run gave a correct answer, so neither cost counts." in both_wrong
    both_right = render.meters_html(costly_right, "naive", fa_data.clean_trace(make_trace()), "oracle")
    assert "fa-bar-na" not in both_right and "cost does not count" not in both_right


def _results_doc():
    def entry(acc, eff, eff_ok, calls, groups):
        m = {"accuracy": acc, "efficiency": eff, "tool_calls": calls, "latency_s": 7.0, "correct_and_optimal_rate": acc / 2,
             "reward": acc}
        if eff_ok is not None:
            m["efficiency_when_correct"] = eff_ok
        return {"n": 40, "metrics": m, "by_template_group": groups}
    g = {"rl_only": {"n": 10, "accuracy": 0.1}, "sft_seen": {"n": 30, "accuracy": 0.2}}
    return {
        "bootstrap": 2000,
        "metric_info": {"efficiency_when_correct": {"label": "efficiency (correct only)"}},
        "splits": {"test_id": {"models": {
            "zero-shot": entry(0.1, 0.8, 0.7, 2.6, g),
            "zero-shot-stop": entry(0.0, 1.0, None, 1.0, g),  # does nothing: raw efficiency 1.0, no correct answer
            "grpo": entry(0.8, 0.95, 0.99, 2.3, {"rl_only": {"n": 10, "accuracy": 0.5}, "sft_seen": {"n": 30, "accuracy": 0.9}}),
        }}},
        "comparisons": {"test_id": {"baseline": "zero-shot", "pairs": [{"model": "grpo", "n_paired": 40, "metrics": {
            "accuracy": {"baseline_mean": 0.1, "candidate_mean": 0.8, "ci_low": 0.6, "ci_high": 0.8, "verdict": "better",
                         "auditkit_p_value": 0.001},
            "tool_calls": {"baseline_mean": 2.6, "candidate_mean": 2.3, "ci_low": -0.5, "ci_high": 0.0,
                           "verdict": "no clear change", "auditkit_p_value": 0.06}}}]}},
    }


def test_results_chart_defaults_to_task_success_and_orders_honest_metrics(tmp_path):
    f = tmp_path / "results.json"
    f.write_text(json.dumps(_results_doc()))
    v = logic.results_view("all", tmp_path / "no-traces", f)
    assert logic.default_metric(v) == "Task success"
    assert v["metrics"][:4] == ["Task success", "Correct and optimal", "Tool calls", "Latency (s)"]
    assert "Efficiency, correct answers only" in v["metrics"]
    assert v["metrics"][-1] == "Efficiency, all runs (wrong ones too)"  # raw mean efficiency goes last
    assert "Efficiency, all runs (wrong ones too)" not in [h for _, h in v["headers"]]
    chart = logic.chart_html(v, "Efficiency, correct answers only")
    assert "n/a" in chart and "A run with no correct answer has no value." in chart  # zero-shot-stop has no value
    raw = logic.chart_html(v, "Efficiency, all runs (wrong ones too)")
    assert "read it next to task success" in raw


def test_results_headline_says_no_clear_change_and_shows_groups(tmp_path):
    f = tmp_path / "results.json"
    f.write_text(json.dumps(_results_doc()))
    v = logic.results_view("all", tmp_path / "no-traces", f)
    head = v["headline"]
    assert "Seen domains: Zero-shot (base model) to SFT + GRPO, 50 steps" in head and "40 paired tasks" in head
    assert '<span class="fa-judge fa-judge-good">better</span>' in head
    assert '<span class="fa-judge fa-judge-none">no clear change</span>' in head
    assert "+0.600 to +0.800" in head and "AuditKit p = 0.001" in head
    assert "Latency (s)" in head and "No interval was computed" in head  # mean from the file, no interval claimed
    detail = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", v["detail"]))
    assert "Task success by template group" in detail and "RL-only templates (10 tasks)" in detail
    assert "SFT + GRPO, 50 steps 0.500 0.900" in detail
    assert "Key comparisons" in detail and "2,000 resamples" in detail and "+0.600 to +0.800" in detail
    # an extra comparison file (GRPO against SFT) adds its rows; a file with another baseline is not mixed in
    extra = {"comparisons": {"test_id": {"baseline": "distilled", "pairs": [{"model": "grpo", "metrics": {"accuracy": {
        "baseline_mean": 0.78, "candidate_mean": 0.8, "ci_low": -0.01, "ci_high": 0.05, "verdict": "no clear change"}}}]}}}
    e = tmp_path / "vs_sft.json"
    e.write_text(json.dumps(extra))
    v2 = logic.results_view("all", tmp_path / "no-traces", f, {"vs_sft": e, "mixed": tmp_path / "missing.json"})
    assert "What 50 GRPO steps add on top of SFT (distilled to grpo)" in v2["detail"]
    tags_are_own(v2["detail"], {"div", "h3", "p", "table", "thead", "tbody", "tr", "th", "td"})


def test_findings_line_comes_from_the_results_files(tmp_path):
    doc = _results_doc()
    doc["splits"]["test_ood"] = {"models": {"distilled": {"metrics": {"accuracy": 0.46}}}}
    doc["comparisons"]["test_ood"] = {"baseline": "zero-shot", "pairs": [{"model": "grpo", "n_paired": 100, "metrics": {
        "accuracy": {"baseline_mean": 0.08, "candidate_mean": 0.47}}}]}
    f = tmp_path / "results.json"
    f.write_text(json.dumps(doc))
    text = logic.findings_text(f, {})
    assert text.startswith("Result on the held-out database domain (100 tasks)") and "0.080" in text and "0.460" in text
    assert logic.findings_text(tmp_path / "missing.json", {}) == ""
    assert "<script" not in render.hero_html(EVIL).lower()


def test_footer_links_fall_back_to_links_file(tmp_path):
    import app as app_module
    f = tmp_path / "links.json"
    f.write_text(json.dumps({"FRUGAL_LINK_GITHUB": "https://github.com/org/repo", "OTHER": "https://x.org", "FRUGAL_LINK_MODELS": 5}))
    links = dict(app_module.footer_links({}, f))
    assert links["GitHub repository"] == "https://github.com/org/repo" and links["Model collection"] == ""
    env_wins = dict(app_module.footer_links({"FRUGAL_LINK_GITHUB": "https://example.org/a"}, f))
    assert env_wins["GitHub repository"] == "https://example.org/a"
    (tmp_path / "bad.json").write_text("not json")
    assert all(u == "" for _, u in app_module.footer_links({}, tmp_path / "bad.json"))


# ---------------------------------------------------------------------------
# Hugging Face Space build
# ---------------------------------------------------------------------------


def _load_build_space():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_space", ROOT / "space" / "build_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_repo(root: Path) -> Path:
    import shutil
    for name in ("app.py", "fa_backend.py", "fa_data.py", "fa_logic.py", "fa_render.py", "requirements.txt"):
        (root / "app").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "app" / name, root / "app" / name)
    shutil.copytree(ROOT / "frugal_analyst", root / "frugal_analyst", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "data" / "tasks").mkdir(parents=True)
    for name in ("val.jsonl", "test_id.jsonl", "test_ood.jsonl"):
        shutil.copy2(ROOT / "data" / "tasks" / name, root / "data" / "tasks" / name)
    (root / "outputs" / "traces").mkdir(parents=True)
    write_traces(root / "outputs" / "traces" / "grpo_test_id.jsonl", [make_trace()])
    write_traces(root / "outputs" / "traces" / "zero-shot_test_id.jsonl", [make_trace(correct=False)])
    (root / "outputs" / "traces" / "grpo_test_id.run.json").write_text("{}")  # not a trace file: left out
    (root / "outputs" / "traces" / "Bad Name_test_id.jsonl").write_text("")
    (root / "outputs" / "eval").mkdir(parents=True)
    (root / "outputs" / "eval" / "results.json").write_text(json.dumps(_results_doc()))
    (root / "data" / "dbs").mkdir()
    (root / "data" / "dbs" / "retail.sqlite").write_text("x")  # databases are never copied
    return root


def _front_matter(text: str) -> dict:
    assert text.startswith("---\n")
    block = text.split("---\n")[1]
    out = {}
    for line in block.strip().splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip('"')
    return out


def test_build_space_writes_an_uploadable_folder(tmp_path):
    import gradio
    bs = _load_build_space()
    repo = _fake_repo(tmp_path / "repo")
    out = tmp_path / "build"
    files = bs.build(out, {"FRUGAL_LINK_GITHUB": "https://github.com/org/repo"}, repo=repo)
    expected = {
        "README.md", "requirements.txt", "app/links.json", bs.MARKER,
        "app/app.py", "app/fa_backend.py", "app/fa_data.py", "app/fa_logic.py", "app/fa_render.py",
        "frugal_analyst/__init__.py", "frugal_analyst/db.py", "frugal_analyst/env.py",
        "data/tasks/val.jsonl", "data/tasks/test_id.jsonl", "data/tasks/test_ood.jsonl",
        "outputs/traces/grpo_test_id.jsonl", "outputs/traces/zero-shot_test_id.jsonl", "outputs/eval/results.json",
    }
    assert expected <= set(files)
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert on_disk == set(files)
    assert not any(f.endswith(".sqlite") or "dbs/" in f or f.endswith(".run.json") or "Bad Name" in f for f in on_disk)
    fm = _front_matter((out / "README.md").read_text(encoding="utf-8"))
    assert fm["sdk"] == "gradio" and fm["sdk_version"] == gradio.__version__
    assert fm["app_file"] == "app/app.py" and (out / fm["app_file"]).is_file()
    assert fm["pinned"] == "false" and fm["title"] and 0 < len(fm["short_description"]) <= 60
    assert "license" not in fm
    reqs = (out / "requirements.txt").read_text(encoding="utf-8")
    assert "gradio" not in [l.split("=")[0] for l in reqs.splitlines() if l and not l.startswith("#")]
    assert json.loads((out / "app" / "links.json").read_text()) == {"FRUGAL_LINK_GITHUB": "https://github.com/org/repo"}
    assert "—" not in (out / "README.md").read_text(encoding="utf-8")
    # a second build replaces the old one; a folder that is not a build is never deleted
    assert bs.build(out, {}, repo=repo) and json.loads((out / "app" / "links.json").read_text()) == {}
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("x")
    with pytest.raises(bs.BuildError):
        bs.build(other, {}, repo=repo)
    assert (other / "keep.txt").is_file()
    with pytest.raises(bs.BuildError):
        bs.build(tmp_path / "b2", {"FRUGAL_LINK_WRITEUP": "javascript:alert(1)"}, repo=repo)
    with pytest.raises(bs.BuildError):
        bs.build(repo, {}, repo=repo)


def test_build_space_cli_reports_missing_inputs(tmp_path, capsys):
    bs = _load_build_space()
    repo = _fake_repo(tmp_path / "repo")
    (repo / "outputs" / "eval" / "results.json").unlink()
    with pytest.raises(bs.BuildError):
        bs.build(tmp_path / "out", {}, repo=repo)
    assert bs.main(["--out", str(tmp_path / "out2"), "--github", "http://not-https.org"]) == 2
    assert "not a plain https URL" in capsys.readouterr().err
