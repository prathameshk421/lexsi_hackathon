"""HTML builders. Every piece of text from a trace, a model or a database goes through `esc`.

Only numbers, fixed words and fixed class names are written into HTML without escaping.
SQL colour is added after escaping: the raw SQL is cut into tokens, each token is escaped,
and a fixed class name is put around keywords, strings and numbers.
"""

from __future__ import annotations

import html
import re

try:
    from fa_data import LEVEL_TEXT, SCRIPTED_LABELS, display_label
except ImportError:  # pragma: no cover
    from .fa_data import LEVEL_TEXT, SCRIPTED_LABELS, display_label

MAX_SQL_SHOWN = 1500
MAX_OUTPUT_LINES = 6
MAX_OUTPUT_CHARS = 520
MAX_OUTPUT_FULL = 6000
MAX_ANSWER_SHOWN = 300
MAX_QUESTION_SHOWN = 400
MAX_TABLE_CELL = 200


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def clip(text, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def num(n) -> str:
    return f"{int(n):,}"


def flt(x, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}"


def times(high: float, low: float) -> str:
    """'2.5x' or '2,901x'. Caller makes sure low > 0."""
    r = high / low
    return f"{r:.1f}x" if r < 10 else f"{r:,.0f}x"


def join_words(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# ---------------------------------------------------------------------------
# SQL display
# ---------------------------------------------------------------------------

_QUOTED = re.compile(r"""('(?:[^']|'')*'|"(?:[^"]|"")*")""")
_BREAK = re.compile(
    r"\s+(?=(?:LEFT\s+JOIN|INNER\s+JOIN|CROSS\s+JOIN|JOIN|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION)\b)",
    re.IGNORECASE,
)


def format_sql(sql) -> str:
    """Put major clauses on their own line. Only whitespace changes, so the meaning is the same."""
    parts = _QUOTED.split(str(sql).strip())
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # quoted text: leave as is
            out.append(part)
        else:
            out.append(_BREAK.sub("\n", re.sub(r"[ \t\r\n]+", " ", part)))
    text = "".join(out).strip()
    return clip(text, MAX_SQL_SHOWN)


SQL_KEYWORDS = frozenset(
    """SELECT FROM WHERE JOIN LEFT RIGHT INNER OUTER CROSS ON AND OR NOT IN IS NULL AS GROUP BY ORDER HAVING
    LIMIT OFFSET DISTINCT UNION ALL WITH CASE WHEN THEN ELSE END ASC DESC BETWEEN LIKE EXISTS RECURSIVE
    INSERT UPDATE DELETE DROP CREATE ALTER PRAGMA ATTACH REPLACE INTO VALUES SET TABLE DATABASE VACUUM""".split()
)
SQL_FUNCS = frozenset(
    "COUNT SUM AVG MIN MAX ROUND STRFTIME DATE JULIANDAY SUBSTR LOWER UPPER COALESCE CAST IFNULL LENGTH TOTAL ABS".split()
)
_SQL_TOKEN = re.compile(
    r"(?P<str>'(?:[^']|'')*'?)|(?P<qid>\"(?:[^\"]|\"\")*\"?)|(?P<cmt>--[^\n]*)"
    r"|(?P<num>\b\d+(?:\.\d+)?\b)|(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<other>.)",
    re.S,
)


def highlight_sql(sql: str) -> str:
    """Escaped SQL with fixed span classes. The input is raw text. Each token is escaped on its own."""
    out = []
    for m in _SQL_TOKEN.finditer(str(sql)):
        kind, text = m.lastgroup, m.group()
        safe = esc(text)
        if kind == "str":
            out.append(f'<span class="sq-s">{safe}</span>')
        elif kind == "num":
            out.append(f'<span class="sq-n">{safe}</span>')
        elif kind == "cmt":
            out.append(f'<span class="sq-c">{safe}</span>')
        elif kind == "word" and text.upper() in SQL_KEYWORDS:
            out.append(f'<span class="sq-k">{safe}</span>')
        elif kind == "word" and text.upper() in SQL_FUNCS:
            out.append(f'<span class="sq-f">{safe}</span>')
        else:
            out.append(safe)
    return "".join(out)


def sql_block(sql) -> str:
    return f'<pre class="fa-code"><code>{highlight_sql(format_sql(sql))}</code></pre>'


def preview_output(text: str) -> tuple[str, int]:
    """First lines of a tool output, and how many lines were left out."""
    lines = str(text).splitlines() or [""]
    shown, used = [], 0
    for line in lines[:MAX_OUTPUT_LINES]:
        room = MAX_OUTPUT_CHARS - used
        if room <= 0:
            break
        shown.append(line if len(line) <= room else line[: max(room - 3, 0)] + "...")
        used += len(line) + 1
    return "\n".join(shown), max(len(lines) - len(shown), 0)


def output_block(text: str) -> str:
    """A short preview. When the output is longer, a native disclosure shows all of it (no script)."""
    preview, hidden = preview_output(text)
    block = f'<pre class="fa-out"><code>{esc(preview)}</code></pre>'
    full = str(text)
    if hidden or len(full) > len(preview) + 1:
        total = len(full.splitlines())
        block += (
            f'<details class="fa-expand"><summary>Show full output ({num(total)} lines)</summary>'
            f'<pre class="fa-out fa-out-full"><code>{esc(clip(full, MAX_OUTPUT_FULL))}</code></pre></details>'
        )
    return block


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------


def notice_html(message: str, kind: str = "info", title: str | None = None) -> str:
    kind = kind if kind in ("info", "warn", "bad") else "info"
    role = "alert" if kind == "bad" else "status"
    head = f'<div class="fa-notice-title">{esc(title)}</div>' if title else ""
    return f'<div class="fa fa-notice fa-notice-{kind}" role="{role}">{head}<div>{esc(message)}</div></div>'


# ---------------------------------------------------------------------------
# One step, one side
# ---------------------------------------------------------------------------


def step_html(index: int, step: dict, enter: bool = False) -> str:
    if step["type"] != "tool_call":
        return ""
    ok = step["ok"]
    meta = [f'{num(step["n_rows"])} rows']
    if step["name"] == "run_sql":
        meta.append(f'{num(step["vm_steps"])} VM steps')
    status = "" if ok else '<span class="fa-status-bad">failed</span>'
    args = step["arguments"]
    db = args.get("database")
    target = f'<span class="fa-db">{esc(clip(db, 40))}</span>' if db is not None else ""
    body = sql_block(args.get("query")) if step["name"] == "run_sql" and "query" in args else ""
    cls = "fa-step" + ("" if ok else " fa-step-bad") + (" fa-enter" if enter else "")
    return (
        f'<li class="{cls}"><span class="fa-idx" aria-hidden="true">{index}</span><div class="fa-step-main">'
        f'<div class="fa-step-head"><span class="fa-tool">{esc(clip(step["name"], 40))}</span>{target}'
        f'<span class="fa-step-meta">{status}<span>{" / ".join(meta)}</span></span></div>'
        f"{body}{output_block(step['output'])}</div></li>"
    )


def _verdict(trace: dict, has_gold: bool) -> str:
    if not has_gold:
        return '<span class="fa-verdict fa-verdict-none">No gold answer, so no verdict</span>'
    if trace["correct"]:
        return '<span class="fa-verdict fa-verdict-good">Correct</span>'
    if not trace["answered"]:
        return '<span class="fa-verdict fa-verdict-bad">No answer</span>'
    if not trace["grounded"]:
        return '<span class="fa-verdict fa-verdict-bad">Wrong: no successful query</span>'
    return '<span class="fa-verdict fa-verdict-bad">Wrong</span>'


def final_html(trace: dict, has_gold: bool) -> str:
    finals = [s for s in trace["steps"] if s["type"] == "final"]
    if finals:
        text = clip(finals[-1]["text"], MAX_ANSWER_SHOWN)
        said = f'<pre class="fa-final-text"><code>{esc(text)}</code></pre>'
    else:
        said = '<div class="fa-muted">The run ended without a final answer.</div>'
    extra = ""
    if has_gold and not trace["correct"]:
        extra = f'<div class="fa-expected">Expected <b>{esc(clip(trace["gold_answer"], 80))}</b></div>'
    return f'<div class="fa-final fa-enter-soft"><div class="fa-label">Final answer</div>{said}{extra}</div>'


def summary_html(trace: dict, has_gold: bool = True) -> str:
    cost, opt = trace["cost"], trace["optimal"]

    def cell(name: str, value: str, ref: int | None, over: bool = False) -> str:
        hint = f'<div class="fa-cell-ref">optimal {num(ref)}</div>' if (has_gold and ref is not None) else ""
        cls = "fa-cell fa-cell-over" if over else "fa-cell"
        return f'<div class="{cls}"><div class="fa-cell-name">{name}</div><div class="fa-cell-val">{value}</div>{hint}</div>'

    def over(value: int, ref) -> bool:
        return has_gold and ref is not None and value > ref * 1.05

    cells = [
        cell("Tool calls", num(cost["tool_calls"]), opt["tool_calls"], over(cost["tool_calls"], opt["tool_calls"])),
        cell("Rows returned", num(cost["rows_returned"]), opt["rows"], over(cost["rows_returned"], opt["rows"])),
        cell("VM steps", num(cost["vm_steps"]), opt["vm_steps"], over(cost["vm_steps"], opt["vm_steps"])),
    ]
    if has_gold:
        cells += [
            cell("Efficiency", flt(trace["efficiency"]), None),
            cell("Reward", flt(trace["reward"]), None),
        ]
    else:
        cells.append(cell("Failed calls", num(cost["failed_calls"]), None))
    return f'<div class="fa-summary">{"".join(cells)}</div>'


def kind_tag(label: str) -> str:
    if label in SCRIPTED_LABELS:
        return '<span class="fa-tag fa-tag-scripted" title="A fixed script, not a trained model">scripted reference</span>'
    return ""


def trajectory_html(trace: dict, label: str, side: str = "a", upto: int | None = None,
                    has_gold: bool = True, show_end: bool = True, animate_last: bool = False) -> str:
    """One run as a card. `upto` shows only the first N tool steps (used while a live run streams)."""
    calls = [s for s in trace["steps"] if s["type"] == "tool_call"]
    shown = calls if upto is None else calls[:upto]
    items = "".join(step_html(i, s, enter=animate_last and i == len(shown)) for i, s in enumerate(shown, 1))
    if not calls:
        steps = '<div class="fa-muted fa-pad">No tool calls in this run.</div>'
    elif items:
        steps = f'<ol class="fa-steps">{items}</ol>'
    else:
        steps = '<div class="fa-muted fa-pad">Waiting for the first tool call.</div>'
    end = final_html(trace, has_gold) + summary_html(trace, has_gold) if show_end else ""
    verdict = _verdict(trace, has_gold) if show_end else ""
    scripted = " fa-card-scripted" if label in SCRIPTED_LABELS else ""
    return (
        f'<section class="fa-card fa-side-{esc(side)}{scripted}">'
        f'<header class="fa-card-head"><div class="fa-card-title"><span class="fa-model">{esc(clip(label, 40))}</span>'
        f"{kind_tag(label)}</div>{verdict}</header>"
        f'<div class="fa-card-body">{steps}{end}</div></section>'
    )


# ---------------------------------------------------------------------------
# Cost meters and the one-line story
# ---------------------------------------------------------------------------


def _ratio_text(value: int, ref: int | None) -> str:
    if not ref:
        return ""
    r = value / ref
    return "at optimal" if abs(r - 1) < 0.05 else f"{times(value, ref)} optimal"


def _bar_class(value: int, ref: int | None) -> str:
    """Neutral bar at or near the optimal cost, warm hatched bar above it."""
    return "fa-bar-ok" if ref is None or value <= ref * 1.05 else "fa-bar-warn"


METER_SPEC = [
    ("Tool calls", "tool_calls", "tool_calls", "Each call is one more model turn."),
    ("Rows returned", "rows_returned", "rows", "Rows sent back into the model context."),
    ("VM steps", "vm_steps", "vm_steps", "SQLite engine instructions. The same on every run."),
]

GLOSSARY_HTML = (
    '<details class="fa-glossary"><summary>How the score works</summary><dl>'
    "<dt>Optimal</dt><dd>The cost of the efficient scripted path (the oracle) for the same task.</dd>"
    "<dt>Efficiency</dt><dd>From 0 to 1. The mean of three ratios against optimal: calls, rows and VM steps.</dd>"
    "<dt>Reward</dt><dd>A correct answer earns 0.5 plus half of its efficiency. A wrong answer earns little. "
    "A correct answer needs at least one successful query, so a guess without a query does not pay.</dd>"
    "<dt>VM steps</dt><dd>SQLite counts the virtual machine instructions a query uses. The count does not change "
    "between runs, so it is a fair measure of engine work. Wall time is not.</dd></dl></details>"
)


def meters_html(ta: dict, la: str, tb: dict, lb: str) -> str:
    rows = []
    for title, ck, ok_, caption in METER_SPEC:
        va, vb = ta["cost"][ck], tb["cost"][ck]
        ref = ta["optimal"].get(ok_)
        scale = max(va, vb, ref or 0, 1)
        lines = []
        for side, label, v in (("a", la, va), ("b", lb, vb)):
            width = max(v / scale * 100, 1.5 if v else 0)
            lines.append(
                f'<div class="fa-line"><span class="fa-line-label" title="{esc(label)}">{esc(clip(label, 24))}</span>'
                f'<div class="fa-track"><div class="fa-bar {_bar_class(v, ref)}" style="width:{width:.1f}%"></div>'
                + (f'<span class="fa-tick" style="left:{ref / scale * 100:.1f}%"></span>' if ref else "")
                + f'</div><span class="fa-line-val">{num(v)}</span>'
                f'<span class="fa-line-ratio">{esc(_ratio_text(v, ref))}</span></div>'
            )
        ref_txt = f"optimal {num(ref)}" if ref is not None else ""
        rows.append(
            f'<div class="fa-meter"><div class="fa-meter-title"><span>{title}</span>'
            f'<span class="fa-meter-opt">{ref_txt}</span></div><p class="fa-meter-cap">{caption}</p>{"".join(lines)}</div>'
        )
    legend = (
        '<div class="fa-legend"><span class="fa-legend-item"><span class="fa-swatch fa-sw-ok"></span>at optimal</span>'
        '<span class="fa-legend-item"><span class="fa-swatch fa-sw-warn"></span>above optimal</span>'
        '<span class="fa-legend-item"><span class="fa-swatch-tick"></span>optimal cost</span></div>'
    )
    return (
        '<div class="fa-panel fa-meters-panel"><div class="fa-panel-head"><h3 class="fa-panel-title">Cost at a glance</h3>'
        f'{legend}</div><div class="fa-meters">{"".join(rows)}</div>{GLOSSARY_HTML}</div>'
    )


def story_html(ta: dict, la: str, tb: dict, lb: str) -> str:
    """Two short sentences, computed from the traces: who is right, and who paid more."""
    na, nb = f"<b>{esc(display_label(la))}</b>", f"<b>{esc(display_label(lb))}</b>"
    ca, cb = ta["correct"], tb["correct"]
    if ca and cb:
        verdict = "Both answers are correct."
    elif ca or cb:
        verdict = f"Only {na if ca else nb} is correct."
    else:
        verdict = "Neither answer is correct."
    spec = [("tool_calls", "tool calls"), ("rows_returned", "rows"), ("vm_steps", "VM steps")]

    def key(t: dict) -> tuple:
        return (t["cost"]["vm_steps"], t["cost"]["rows_returned"], t["cost"]["tool_calls"])

    if key(ta) == key(tb):
        cost = "Both runs cost the same."
    else:
        (hi, nhi), (lo, nlo) = ((ta, na), (tb, nb)) if key(ta) > key(tb) else ((tb, nb), (ta, na))
        more, less = [], []
        for k, word in spec:
            vh, vl = hi["cost"][k], lo["cost"][k]
            if vh > vl:
                more.append(f'<span class="fa-num">{times(vh, vl)}</span> the {word}' if vl > 0
                            else f'<span class="fa-num">{num(vh)}</span> {word} against none')
            elif vl > vh:
                less.append(word)
        cost = f"{nhi} used {join_words(more)} of {nlo}." if more else "Both runs cost about the same."
        if less:
            cost += f" {nlo} used more {join_words(less)}."
    return f'<div class="fa-story"><p class="fa-story-verdict">{verdict}</p><p class="fa-story-cost">{cost}</p></div>'


# ---------------------------------------------------------------------------
# Replay and live pages
# ---------------------------------------------------------------------------


def question_html(trace: dict, show_gold: bool = True) -> str:
    level = trace["level"]
    meta = [f'<span class="fa-mono">{esc(trace["task_id"])}</span>']
    if trace["domain"]:
        meta.append(f"<span>{esc(trace['domain'])}</span>")
    if level in LEVEL_TEXT:
        meta.append(f'<span>{esc(LEVEL_TEXT[level])}</span>')
    elif level:
        meta.append(f"<span>level {esc(level)}</span>")
    gold = (
        f'<div class="fa-gold">Gold answer <b class="fa-mono">{esc(clip(trace["gold_answer"], 80))}</b></div>'
        if show_gold else ""
    )
    return (
        f'<div class="fa-question"><div class="fa-meta">{"".join(meta)}</div>'
        f'<h2 class="fa-q">{esc(clip(trace["question"], MAX_QUESTION_SHOWN))}</h2>{gold}</div>'
    )


def replay_html(ta: dict, la: str, tb: dict, lb: str) -> str:
    return (
        '<div class="fa fa-replay">'
        f'<div class="fa-lead">{question_html(ta)}{story_html(ta, la, tb, lb)}</div>'
        f"{meters_html(ta, la, tb, lb)}"
        f'<div class="fa-cols">{trajectory_html(ta, la, "a")}{trajectory_html(tb, lb, "b")}</div></div>'
    )


def live_html(trace: dict, label: str, upto: int, done: bool, has_gold: bool, question: str, note: str = "") -> str:
    head = (
        question_html(trace, show_gold=has_gold)
        if has_gold
        else '<div class="fa-question"><div class="fa-meta"><span>typed question</span></div>'
        f'<h2 class="fa-q">{esc(clip(question, MAX_QUESTION_SHOWN))}</h2></div>'
    )
    note_html = f'<div class="fa-note">{esc(note)}</div>' if note else ""
    total = len([s for s in trace["steps"] if s["type"] == "tool_call"])
    status = (
        "" if done else
        f'<div class="fa-running" role="status"><span class="fa-spinner" aria-hidden="true"></span>'
        f"Running... step {num(min(upto + 1, max(total, 1)))} of {num(max(total, 1))}</div>"
    )
    return (
        f'<div class="fa fa-live">{head}{note_html}'
        f'{trajectory_html(trace, label, "a", upto=upto, has_gold=has_gold, show_end=done, animate_last=not done)}'
        f"{status}</div>"
    )


# ---------------------------------------------------------------------------
# Schema, tables, playground
# ---------------------------------------------------------------------------

_SCHEMA_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*--\s*([\d,]+)\s+rows?$")


def schema_html(text: str) -> str:
    """The compact schema text as a list of tables. Lines that do not parse are shown as plain text."""
    items = []
    for line in str(text).splitlines():
        m = _SCHEMA_LINE.match(line.strip())
        if m:
            cols = ", ".join(esc(c.strip()) for c in m.group(2).split(","))
            items.append(
                f'<li><div class="fa-schema-head"><span class="fa-mono fa-schema-name">{esc(m.group(1))}</span>'
                f'<span class="fa-schema-rows">{num(m.group(3).replace(",", ""))} rows</span></div>'
                f'<div class="fa-schema-cols fa-mono">{cols}</div></li>'
            )
        elif line.strip():
            items.append(f'<li><div class="fa-schema-cols fa-mono">{esc(line)}</div></li>')
    if not items:
        return notice_html(str(text) or "No schema.", "info")
    n = len(items)
    return (
        f'<div class="fa"><details class="fa-schema-box" open><summary>Schema: {n} table{"s" if n != 1 else ""}</summary>'
        f'<ul class="fa-schema">{"".join(items)}</ul></details></div>'
    )


_NUMERIC = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")
_CLASS_OK = re.compile(r"^[a-z0-9 -]{0,80}$")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def table_html(headers: list, rows: list[list], row_classes: list[str] | None = None,
               empty: str = "No rows.", label: str = "Table") -> str:
    """A plain, scannable table. Numeric columns are right aligned with tabular figures."""
    if not headers or not rows:
        return f'<div class="fa"><div class="fa-empty">{esc(empty)}</div></div>'
    ncol = len(headers)
    numeric, decimals = [], []
    for j in range(ncol):
        vals = [r[j] for r in rows if j < len(r) and r[j] not in ("", None)]
        is_num = bool(vals) and all(_is_number(v) or (isinstance(v, str) and _NUMERIC.match(v)) for v in vals)
        numeric.append(is_num)
        floats = [v for v in vals if isinstance(v, float)]
        if not floats:
            decimals.append(None)
        else:
            top = max(abs(v) for v in floats)
            decimals.append(3 if top < 10 else 2 if top < 1000 else 0)  # one precision per column, so digits line up

    def cell(v, j) -> str:
        if isinstance(v, float) and _is_number(v):
            text = f"{v:,.{decimals[j] if decimals[j] is not None else 3}f}"
        elif _is_number(v):
            text = f"{v:,}"
        else:
            text = clip("" if v is None else v, MAX_TABLE_CELL)
        return esc(text)

    head = "".join(
        f'<th scope="col" class="{"fa-num" if numeric[j] else ""}">{esc(clip(h, 80))}</th>' for j, h in enumerate(headers)
    )
    body = []
    for i, r in enumerate(rows):
        cls = (row_classes[i] if row_classes and i < len(row_classes) else "") or ""
        cls = cls if isinstance(cls, str) and _CLASS_OK.match(cls) else ""  # only fixed class names
        cells = "".join(
            f'<td class="{"fa-num" if numeric[j] else ""}">{cell(r[j] if j < len(r) else "", j)}</td>' for j in range(ncol)
        )
        body.append(f'<tr class="{esc(cls)}">{cells}</tr>')
    return (
        f'<div class="fa"><div class="fa-table-wrap" role="region" aria-label="{esc(label)}" tabindex="0">'
        f'<table class="fa-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>'
    )


def query_stats_html(n_rows: int, vm_steps: int, capped: bool, hint: str, error: str | None = None, note: str = "") -> str:
    if error:
        return (
            '<div class="fa fa-notice fa-notice-bad" role="alert"><div class="fa-notice-title">Query refused or failed</div>'
            f'<div class="fa-mono fa-error-text">{esc(clip(error, 400))}</div>'
            f'<div class="fa-muted">Cost so far: {num(vm_steps)} VM steps</div></div>'
        )
    rows = f"{num(n_rows)}{'+' if capped else ''}"
    return (
        '<div class="fa"><div class="fa-summary fa-summary-2">'
        f'<div class="fa-cell"><div class="fa-cell-name">Rows returned</div><div class="fa-cell-val">{rows}</div></div>'
        f'<div class="fa-cell"><div class="fa-cell-name">VM steps</div><div class="fa-cell-val">{num(vm_steps)}</div></div>'
        f"</div>"
        + (f'<div class="fa-hint">{esc(hint)}</div>' if hint else "")
        + (f'<div class="fa-note">{esc(note)}</div>' if note else "")
        + "</div>"
    )


def pair_html(items: list[tuple[str, str, int, int, str | None]]) -> str:
    """items: (title, sql, rows, vm_steps, error). Shows cheap and costly side by side with shared scales."""
    max_rows = max([i[2] for i in items] + [1])
    max_vm = max([i[3] for i in items] + [1])
    cards = []
    ref_rows, ref_vm = (items[0][2], items[0][3]) if items else (0, 0)
    for n, (title, sql, rows, vm, err) in enumerate(items):
        if err:
            stats = f'<div class="fa-notice fa-notice-bad" role="alert">{esc(clip(err, 300))}</div>'
        else:
            stats = "".join(
                f'<div class="fa-line"><span class="fa-line-label">{name}</span><div class="fa-track">'
                f'<div class="fa-bar {"fa-bar-ok" if n == 0 else _bar_class(v, ref)}" style="width:{max(v / m * 100, 1.5 if v else 0):.1f}%"></div></div>'
                f'<span class="fa-line-val">{num(v)}</span>'
                + (f'<span class="fa-line-ratio">{esc(_ratio_text(v, ref).replace("optimal", "of the cheap query"))}</span>' if n else "")
                + "</div>"
                for name, v, m, ref in (("Rows", rows, max_rows, ref_rows), ("VM steps", vm, max_vm, ref_vm))
            )
        cards.append(
            f'<section class="fa-card fa-pair-{"cheap" if n == 0 else "costly"}"><header class="fa-card-head">'
            f'<div class="fa-card-title"><span class="fa-model">{esc(title)}</span></div></header>'
            f'<div class="fa-card-body">{sql_block(sql)}<div class="fa-pair-stats">{stats}</div></div></section>'
        )
    return f'<div class="fa fa-enter-soft"><div class="fa-cols">{"".join(cards)}</div></div>'


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def pretty_key(key: str) -> str:
    text = str(key).replace("_", " ")
    return text[:1].upper() + text[1:]


_LOWER_BETTER = ("tool call", "tool_call", "rows", "vm step", "vm_step", "latency", "failed call", "failed_call")
_HIGHER_BETTER = ("accuracy", "reward", "efficiency", "rate", "validity", "0 failed")


def better_direction(name: str) -> str | None:
    """'up' or 'down' when the metric name says which way is better, else None."""
    n = str(name).lower()
    if any(w in n for w in _HIGHER_BETTER):
        return "up"
    if any(w in n for w in _LOWER_BETTER):
        return "down"
    return None


def headline_html(before_label: str, after_label: str, cards: list[tuple[str, float, float]], source: str) -> str:
    if not cards:
        return ""
    out = []
    for card in cards:
        name, b, a = card[:3]
        extra = card[3] if len(card) > 3 else ""
        delta = a - b
        sign = "+" if delta > 0 else ""
        way = better_direction(name)
        judge = ""
        if way and delta != 0:
            good = (delta > 0) == (way == "up")
            judge = f'<span class="fa-judge {"fa-judge-good" if good else "fa-judge-bad"}">{"better" if good else "worse"}</span>'
        extra_html = f'<div class="fa-cell-ref">{esc(extra)}</div>' if extra else ""
        out.append(
            f'<div class="fa-stat"><div class="fa-cell-name">{esc(pretty_key(name))}</div>'
            f'<div class="fa-stat-vals"><span class="fa-stat-after">{esc(_fmt_value(a))}</span>'
            f'<span class="fa-stat-before"><span class="fa-arrow">from</span> {esc(_fmt_value(b))}</span></div>'
            f'<div class="fa-stat-change">change {esc(sign + _fmt_value(round(delta, 4)))} {judge}</div>{extra_html}</div>'
        )
    return (
        f'<div class="fa fa-headline"><div class="fa-panel-head"><h3 class="fa-panel-title">{esc(before_label)} to {esc(after_label)}</h3>'
        f'<span class="fa-muted">{esc(source)}</span></div><div class="fa-stats">{"".join(out)}</div></div>'
    )


def _tick(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.2f}".rstrip("0").rstrip(".")


def chart_html(metric: str, items: list[tuple[str, float, str, bool]], upper: float) -> str:
    """Horizontal bars on an axis that starts at 0. items: (name, value, group, is_scripted)."""
    groups: list[str] = []
    for _, _, g, _ in items:
        if g not in groups:
            groups.append(g)
    legend = "".join(
        f'<span class="fa-legend-item"><span class="fa-swatch fa-g{i % 4}"></span>{esc(g)}</span>' for i, g in enumerate(groups)
    )
    rows = []
    for name, value, group, scripted in items:
        width = max(value / upper * 100, 1.0 if value > 0 else 0.0)
        cls = f"fa-g{groups.index(group) % 4}" + (" fa-hatch" if scripted else "")
        rows.append(
            f'<div class="fa-crow"><span class="fa-crow-label" title="{esc(name)}">{esc(clip(name, 40))}</span>'
            f'<div class="fa-track"><div class="fa-bar {cls}" style="width:{width:.1f}%"></div></div>'
            f'<span class="fa-line-val">{esc(_tick(value))}</span></div>'
        )
    axis = (
        '<div class="fa-crow fa-axis"><span></span><div class="fa-axis-ticks">'
        f"<span>0</span><span>{_tick(upper / 2)}</span><span>{_tick(upper)}</span></div><span></span></div>"
    )
    return (
        f'<div class="fa fa-panel fa-chart"><div class="fa-panel-head"><h3 class="fa-panel-title">{esc(metric)}</h3>'
        f'<div class="fa-legend">{legend}<span class="fa-legend-item"><span class="fa-swatch fa-hatch fa-g-none"></span>'
        f"hatched: scripted reference policy</span></div></div>{''.join(rows)}{axis}</div>"
    )


# ---------------------------------------------------------------------------
# Page frame: header, hero and footer
# ---------------------------------------------------------------------------

_SAFE_URL = re.compile(r"^https://[A-Za-z0-9.-]+(?::\d{1,5})?(?:/[^\s\"'<>\\`]*)?$")


def safe_url(url) -> str | None:
    """An https URL with no quotes, spaces or angle brackets, else None."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    return url if len(url) <= 300 and _SAFE_URL.match(url) else None


MARK_SVG = (
    '<svg class="fa-mark" viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false">'
    '<rect x="2" y="3" width="16" height="3" rx="1.5"></rect><rect x="2" y="8.5" width="10" height="3" rx="1.5"></rect>'
    '<rect x="2" y="14" width="4" height="3" rx="1.5"></rect></svg>'
)


def header_html(event: str) -> str:
    return (
        '<div class="fa fa-top"><div class="fa-brand">' + MARK_SVG + '<span>Frugal Analyst</span></div>'
        f'<div class="fa-top-meta">{esc(event)}</div></div>'
    )


def hero_html() -> str:
    return (
        '<div class="fa fa-hero"><h1>Same question. Same right answer. <span>Very different cost.</span></h1>'
        "<p>Frugal Analyst trains Tiny Aya, a small open model, to answer data questions over SQLite with two tools. "
        "The reward pays for a correct answer, and for few tool calls, few rows pulled into context and little "
        "database work.</p></div>"
    )


def footer_html(event: str, base_model: str, links: list[tuple[str, str]]) -> str:
    """links: (text, url). Only safe https URLs are shown. Unset links are left out."""
    shown = [(t, u) for t, u in ((t, safe_url(u)) for t, u in links) if u]
    nav = "".join(
        f'<a href="{esc(u)}" target="_blank" rel="noopener noreferrer">{esc(clip(t, 40))}</a>' for t, u in shown
    )
    return (
        '<footer class="fa fa-footer"><div class="fa-footer-main"><div class="fa-brand">' + MARK_SVG
        + '<span>Frugal Analyst</span></div>'
        f'<div class="fa-footer-meta"><span>{esc(event)}</span><span>Base model <span class="fa-mono">{esc(clip(base_model, 80))}</span></span></div></div>'
        + (f'<nav class="fa-footer-links" aria-label="Project links">{nav}</nav>' if nav else "")
        + "</footer>"
    )


# ---------------------------------------------------------------------------
# CSS for the HTML above. Colour tokens follow the page theme (light, or .dark on body).
# ---------------------------------------------------------------------------

CSS = """
:root{--fa-bg:#fafaf9;--fa-surface:#ffffff;--fa-sunken:#f4f4f2;--fa-line:#e4e4e0;--fa-line-2:#d2d2cd;
  --fa-text:#16181d;--fa-text-2:#575b63;--fa-text-3:#6b7078;--fa-accent:#2b59d9;--fa-accent-soft:#ebf0fd;
  --fa-ok:#7c828c;--fa-over:#c2410c;--fa-over-soft:#fdf1ea;--fa-good:#1a7f37;--fa-bad:#c62828;--fa-warn:#a15c00;
  --fa-sq-k:#7a3fc2;--fa-sq-s:#a3470a;--fa-sq-n:#0b6e78;--fa-sq-f:#2b59d9;--fa-sq-c:#6b7078;
  --fa-g0:#2b59d9;--fa-g1:#5d6470;--fa-g2:#0b7a6e;--fa-g3:#a15c00;
  --fa-sans:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --fa-mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --fa-r-sm:4px;--fa-r:6px;--fa-r-lg:10px;color-scheme:light}
.dark{--fa-bg:#0b0c0e;--fa-surface:#131417;--fa-sunken:#1a1c20;--fa-line:#2a2c31;--fa-line-2:#3a3d44;
  --fa-text:#ecedee;--fa-text-2:#a3a7ae;--fa-text-3:#9196a0;--fa-accent:#87a5ff;--fa-accent-soft:rgba(135,165,255,.14);
  --fa-ok:#8b919b;--fa-over:#fb923c;--fa-over-soft:rgba(251,146,60,.12);--fa-good:#4ade80;--fa-bad:#f87171;--fa-warn:#fbbf24;
  --fa-sq-k:#c4a1ff;--fa-sq-s:#f0a868;--fa-sq-n:#5fd3dc;--fa-sq-f:#87a5ff;--fa-sq-c:#9196a0;
  --fa-g0:#87a5ff;--fa-g1:#a3a7ae;--fa-g2:#4fd1c0;--fa-g3:#fbbf24;color-scheme:dark}
.fa{color:var(--fa-text);font-family:var(--fa-sans);font-size:15px;line-height:1.5;max-width:100%;min-width:0}
.fa *,.fa *::before,.fa *::after{box-sizing:border-box}
.fa h1,.fa h2,.fa h3,.fa p,.fa ol,.fa ul,.fa dl,.fa dd{margin:0;padding:0}
.fa b{font-weight:600}
.fa-mono{font-family:var(--fa-mono)}
.fa-num{font-variant-numeric:tabular-nums}
.fa-muted{color:var(--fa-text-2);font-size:13px}
.fa-pad{padding:8px 0}
.fa-label{font-size:12px;font-weight:500;color:var(--fa-text-2);text-transform:uppercase;letter-spacing:.06em}

/* header, hero, footer */
.fa-top{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px 0;border-bottom:1px solid var(--fa-line)}
.fa-brand{display:inline-flex;align-items:center;gap:8px;font-weight:600;font-size:15px;letter-spacing:-.01em}
.fa-mark{fill:var(--fa-text);flex:none}
.fa-mark rect:nth-child(3){fill:var(--fa-accent)}
.fa-top-meta{font-size:13px;color:var(--fa-text-2);text-align:right}
.fa-hero{padding:40px 0 32px;max-width:860px}
.fa-hero h1{font-size:40px;line-height:1.12;font-weight:600;letter-spacing:-.025em;color:var(--fa-text);text-wrap:balance}
.fa-hero h1 span{color:var(--fa-text-2);display:block}
.fa-hero p{margin-top:16px;font-size:17px;line-height:1.55;color:var(--fa-text-2);max-width:66ch}
.fa-footer{margin-top:64px;padding:24px 0 40px;border-top:1px solid var(--fa-line);display:flex;flex-wrap:wrap;gap:16px 32px;
  justify-content:space-between;align-items:flex-start;font-size:13px;color:var(--fa-text-2)}
.fa-footer-main{display:flex;flex-direction:column;gap:8px}
.fa-footer-meta{display:flex;flex-wrap:wrap;gap:4px 24px}
.fa-footer-links{display:flex;flex-wrap:wrap;gap:4px 24px}
.fa-footer-links a{color:var(--fa-text);text-decoration:none;display:inline-flex;align-items:center;min-height:40px}
.fa-footer-links a:hover{color:var(--fa-accent);text-decoration:underline;text-underline-offset:3px}
.fa a:focus-visible,.fa summary:focus-visible,.fa-table-wrap:focus-visible{outline:2px solid var(--fa-accent);outline-offset:2px;border-radius:var(--fa-r-sm)}

/* question and story */
.fa-lead{display:grid;gap:16px;margin:0 0 24px}
.fa-meta{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;color:var(--fa-text-2)}
.fa-meta span+span{position:relative}
.fa-q{font-size:22px;line-height:1.35;font-weight:500;letter-spacing:-.01em;margin:8px 0 0;overflow-wrap:anywhere;max-width:72ch;color:var(--fa-text)}
.fa-gold{margin-top:8px;font-size:14px;color:var(--fa-text-2)}
.fa-gold b{color:var(--fa-text);font-weight:500;margin-left:4px}
.fa-story{border-left:2px solid var(--fa-accent);padding:4px 0 4px 16px;max-width:96ch}
.fa-story-verdict{font-size:16px;font-weight:600}
.fa-story-cost{font-size:16px;color:var(--fa-text-2);margin-top:2px}
.fa-story-cost b{color:var(--fa-text)}
.fa-story-cost .fa-num{color:var(--fa-text);font-weight:600}

/* panels and meters */
.fa-panel{border:1px solid var(--fa-line);border-radius:var(--fa-r-lg);padding:20px 24px;margin:0 0 24px;background:var(--fa-surface)}
.fa-panel-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:8px 24px;margin-bottom:16px}
.fa-panel-title{font-size:15px;font-weight:600;letter-spacing:-.005em}
.fa-meters{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:24px 32px}
.fa-meter-title{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-weight:600;font-size:14px}
.fa-meter-opt{font-size:12px;font-weight:400;color:var(--fa-text-2);font-variant-numeric:tabular-nums}
.fa-meter-cap{font-size:12px;color:var(--fa-text-2);margin:2px 0 8px}
.fa-line{display:grid;grid-template-columns:minmax(56px,96px) minmax(48px,1fr) auto;column-gap:12px;align-items:center;margin:6px 0 0}
.fa-line-label{font-size:12px;color:var(--fa-text-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fa-track{position:relative;height:8px;background:var(--fa-sunken);border-radius:2px}
.fa-bar{height:100%;border-radius:2px;min-width:0}
.fa-bar-ok,.fa-sw-ok{background:var(--fa-ok)}
.fa-bar-warn,.fa-sw-warn{background:repeating-linear-gradient(135deg,var(--fa-over) 0 4px,transparent 4px 6px),var(--fa-over-soft);
  box-shadow:inset 0 0 0 1px var(--fa-over)}
.fa-tick{position:absolute;top:-4px;bottom:-4px;width:2px;margin-left:-1px;background:var(--fa-text);border-radius:1px}
.fa-line-val{font-family:var(--fa-mono);font-variant-numeric:tabular-nums;font-weight:500;font-size:13px;text-align:right;min-width:3ch}
.fa-line-ratio{grid-column:2 / 4;font-size:11px;color:var(--fa-text-2);text-align:right;margin-top:2px;font-variant-numeric:tabular-nums}
.fa-legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--fa-text-2)}
.fa-legend-item{display:inline-flex;align-items:center;gap:6px}
.fa-swatch{width:14px;height:8px;border-radius:2px;display:inline-block}
.fa-swatch-tick{width:2px;height:12px;background:var(--fa-text);display:inline-block;border-radius:1px}
.fa-glossary{margin-top:20px;border-top:1px solid var(--fa-line);padding-top:12px;font-size:13px}
.fa-glossary dl{display:grid;grid-template-columns:max-content 1fr;gap:8px 16px;margin-top:12px;max-width:90ch}
.fa-glossary dt{font-weight:600}
.fa-glossary dd{color:var(--fa-text-2)}
.fa details>summary{cursor:pointer;color:var(--fa-text-2);font-size:13px;list-style:none;display:inline-flex;align-items:center;gap:6px;min-height:32px}
.fa details>summary::-webkit-details-marker{display:none}
.fa details>summary::before{content:"";width:6px;height:6px;border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;
  transform:rotate(-45deg);transition:transform .15s ease;margin-right:2px}
.fa details[open]>summary::before{transform:rotate(45deg)}
.fa details>summary:hover{color:var(--fa-text)}

/* trajectory cards */
.fa-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:24px;align-items:start}
.fa-card{border:1px solid var(--fa-line);border-radius:var(--fa-r-lg);background:var(--fa-surface);min-width:0}
.fa-card-scripted{border-style:dashed;border-color:var(--fa-line-2)}
.fa-card-head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 12px;padding:14px 20px;border-bottom:1px solid var(--fa-line)}
.fa-card-title{display:flex;flex-wrap:wrap;align-items:center;gap:8px;min-width:0}
.fa-model{font-weight:600;font-size:15px;overflow-wrap:anywhere}
.fa-card-body{padding:16px 20px 20px}
.fa-tag{font-size:11px;font-weight:500;padding:1px 6px;border:1px dashed var(--fa-line-2);border-radius:var(--fa-r-sm);color:var(--fa-text-2);white-space:nowrap}
.fa-verdict{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:600}
.fa-verdict::before{content:"";width:8px;height:8px;border-radius:50%;background:currentColor;flex:none}
.fa-verdict-good{color:var(--fa-good)}.fa-verdict-bad{color:var(--fa-bad)}
.fa-verdict-none{color:var(--fa-text-2);font-weight:500}
.fa-verdict-none::before{background:transparent;border:1.5px solid currentColor}
.fa-steps{list-style:none;position:relative}
.fa-steps::before{content:"";position:absolute;left:11px;top:12px;bottom:12px;width:1px;background:var(--fa-line)}
.fa-step{position:relative;display:grid;grid-template-columns:24px minmax(0,1fr);gap:12px;padding:0 0 16px}
.fa-step:last-child{padding-bottom:4px}
.fa-idx{position:relative;z-index:1;display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;
  background:var(--fa-surface);border:1px solid var(--fa-line-2);font-size:12px;font-weight:500;font-family:var(--fa-mono);color:var(--fa-text-2)}
.fa-step-bad .fa-idx{border-color:var(--fa-bad);color:var(--fa-bad)}
.fa-step-main{min-width:0}
.fa-step-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 8px;min-height:24px;margin-bottom:8px}
.fa-tool{font-family:var(--fa-mono);font-weight:500;font-size:13px}
.fa-db{font-size:12px;color:var(--fa-text-2)}
.fa-step-meta{display:inline-flex;flex-wrap:wrap;gap:4px 8px;margin-left:auto;font-size:12px;color:var(--fa-text-2);font-family:var(--fa-mono);font-variant-numeric:tabular-nums}
.fa-status-bad{color:var(--fa-bad);font-weight:600}
.fa pre{display:block;animation:none !important;transition:none !important;content-visibility:visible !important;contain:none !important;opacity:1 !important;margin:0;padding:10px 12px;border-radius:var(--fa-r);overflow-x:auto;max-width:100%;
  font-family:var(--fa-mono);font-size:12.5px;line-height:1.55;white-space:pre;tab-size:2;
  background:var(--fa-sunken);color:var(--fa-text);border:1px solid var(--fa-line)}
.fa pre code{background:none;padding:0;font:inherit;color:inherit;border:0}
.fa-out{max-height:170px;overflow:auto}
.fa-code+.fa-out{margin-top:6px}
.fa-out{background:transparent !important;color:var(--fa-text-2) !important}
.fa-out-full{max-height:420px !important;margin-top:6px !important}
.fa-expand{margin-top:2px}
.sq-k{color:var(--fa-sq-k);font-weight:500}.sq-s{color:var(--fa-sq-s)}.sq-n{color:var(--fa-sq-n)}.sq-f{color:var(--fa-sq-f)}.sq-c{color:var(--fa-sq-c);font-style:italic}
.fa-final{margin-top:8px;padding:14px 16px;border-radius:var(--fa-r);background:var(--fa-sunken)}
.fa-final-text{background:transparent !important;border:0 !important;padding:4px 0 0 !important;white-space:pre-wrap !important;overflow-wrap:anywhere;font-size:14px !important}
.fa-expected{font-size:13px;color:var(--fa-text-2);margin-top:6px}
.fa-expected b{color:var(--fa-text);font-family:var(--fa-mono);font-weight:500;margin-left:4px}
.fa-summary{display:flex;flex-wrap:wrap;gap:1px;margin-top:16px;background:var(--fa-line);
  border:1px solid var(--fa-line);border-radius:var(--fa-r);overflow:hidden}
.fa-summary-2{max-width:420px;margin-top:0}
.fa-cell{flex:1 1 84px;padding:10px 12px;background:var(--fa-surface);min-width:0}
.fa-cell-name{font-size:12px;color:var(--fa-text-2)}
.fa-cell-val{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em;margin-top:2px}
.fa-cell-over .fa-cell-val{color:var(--fa-over)}
.fa-cell-ref{font-size:12px;color:var(--fa-text-2);font-variant-numeric:tabular-nums}

/* notices, hints, live */
.fa-notice{border:1px solid var(--fa-line);border-left:3px solid var(--fa-ok);border-radius:var(--fa-r);padding:12px 16px;background:var(--fa-surface);font-size:14px;color:var(--fa-text)}
.fa-notice-title{font-weight:600;margin-bottom:2px}
.fa-notice-bad{border-left-color:var(--fa-bad)}.fa-notice-warn{border-left-color:var(--fa-warn)}
.fa-notice .fa-muted{margin-top:4px}
.fa-error-text{font-size:13px;overflow-wrap:anywhere}
.fa-hint{margin-top:12px;padding:10px 14px;border-radius:var(--fa-r);background:var(--fa-over-soft);border-left:3px solid var(--fa-over);font-size:14px;max-width:80ch}
.fa-note{margin:8px 0 12px;font-size:13px;color:var(--fa-text-2)}
.fa-empty{padding:24px;border:1px dashed var(--fa-line-2);border-radius:var(--fa-r-lg);color:var(--fa-text-2);font-size:14px;text-align:center}
.fa-live .fa-question{margin-bottom:16px}
.fa-running{display:flex;align-items:center;gap:10px;margin-top:12px;font-size:13px;color:var(--fa-text-2)}
.fa-spinner{width:14px;height:14px;border-radius:50%;border:2px solid var(--fa-line-2);border-top-color:var(--fa-accent);animation:fa-spin .8s linear infinite}
@keyframes fa-spin{to{transform:rotate(360deg)}}
@keyframes fa-in{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@keyframes fa-fade{from{opacity:0}to{opacity:1}}
.fa-enter{animation:fa-in .18s ease-out both}
.fa-enter-soft{animation:fa-fade .2s ease-out both}

/* schema */
.fa-schema{list-style:none;display:grid;gap:8px;margin-top:8px !important}
.fa-schema li{padding:10px 12px;border:1px solid var(--fa-line);border-radius:var(--fa-r);background:var(--fa-surface)}
.fa-schema-head{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.fa-schema-name{font-weight:500;font-size:13px}
.fa-schema-rows{font-size:12px;color:var(--fa-text-2);font-variant-numeric:tabular-nums;white-space:nowrap}
.fa-schema-cols{font-size:12px;color:var(--fa-text-2);margin-top:4px;overflow-wrap:anywhere;line-height:1.5}

/* tables */
.fa-table-wrap{overflow:auto;max-width:100%;max-height:480px;border:1px solid var(--fa-line);border-radius:var(--fa-r-lg);background:var(--fa-surface)}
.fa-table{border-collapse:separate;border-spacing:0;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
.fa .fa-table,.fa .fa-table th,.fa .fa-table td{border:0 !important;margin:0}
.fa .fa-table th,.fa .fa-table td{padding:8px 14px;border-bottom:1px solid var(--fa-line) !important;text-align:left;white-space:nowrap;border-bottom:1px solid var(--fa-line);max-width:320px;overflow:hidden;text-overflow:ellipsis}
.fa-table th{position:sticky;top:0;z-index:1;background:var(--fa-sunken);font-weight:500;font-size:12px;color:var(--fa-text-2)}
.fa .fa-table tbody tr:last-child td{border-bottom:0 !important}
.fa-table tbody tr:hover td{background:var(--fa-sunken)}
.fa-table .fa-num{text-align:right;font-family:var(--fa-mono);font-size:12.5px}
.fa-table td:first-child{font-weight:500}
.fa-table td:first-child,.fa-table th:first-child{position:sticky;left:0;z-index:1;background:var(--fa-surface)}
.fa-table th:first-child{z-index:2;background:var(--fa-sunken)}
.fa-table tr.fa-row-scripted td{color:var(--fa-text-2)}
.fa-table tr.fa-row-scripted td:first-child{font-style:italic}
.fa .fa-table tr.fa-row-first-scripted td{border-top:1px dashed var(--fa-line-2) !important}

/* results */
.fa-headline{margin:0 0 24px}
.fa-stats{display:flex;flex-wrap:wrap;gap:1px;background:var(--fa-line);border:1px solid var(--fa-line);border-radius:var(--fa-r-lg);overflow:hidden}
.fa-stat{flex:1 1 168px;padding:16px;background:var(--fa-surface);min-width:0}
.fa-stat-vals{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-top:4px}
.fa-stat-after{font-size:24px;font-weight:600;letter-spacing:-.015em;font-variant-numeric:tabular-nums}
.fa-stat-before{font-size:13px;color:var(--fa-text-2);font-variant-numeric:tabular-nums}
.fa-arrow{color:var(--fa-text-3)}
.fa-stat-change{font-size:12px;color:var(--fa-text-2);margin-top:4px;font-variant-numeric:tabular-nums}
.fa-judge{font-weight:600;margin-left:4px}
.fa-judge-good{color:var(--fa-good)}.fa-judge-bad{color:var(--fa-bad)}
.fa-chart{margin:0}
.fa-crow{display:grid;grid-template-columns:minmax(80px,min(220px,34%)) minmax(60px,1fr) 52px;column-gap:12px;align-items:center;margin:8px 0}
.fa-crow-label{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fa-crow .fa-track{height:12px}
.fa-axis{margin-top:4px}
.fa-axis-ticks{display:flex;justify-content:space-between;font-size:11px;color:var(--fa-text-2);border-top:1px solid var(--fa-line);padding-top:4px;font-variant-numeric:tabular-nums}
.fa-g0{background-color:var(--fa-g0)}.fa-g1{background-color:var(--fa-g1)}.fa-g2{background-color:var(--fa-g2)}.fa-g3{background-color:var(--fa-g3)}
.fa-g-none{background-color:var(--fa-sunken);border:1px solid var(--fa-line-2)}
.fa-hatch{background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.55) 0 3px,transparent 3px 6px)}
.dark .fa-hatch{background-image:repeating-linear-gradient(135deg,rgba(0,0,0,.45) 0 3px,transparent 3px 6px)}
.fa-pair-stats{margin-top:12px}

@media (max-width:767px){
  .fa-hero{padding:32px 0 24px}.fa-hero h1{font-size:30px}.fa-hero p{font-size:16px}
  .fa-q{font-size:19px}.fa-panel{padding:16px}.fa-card-head{padding:12px 16px}.fa-card-body{padding:14px 16px 16px}
  .fa-glossary dl{grid-template-columns:1fr;gap:2px 0}.fa-glossary dd{margin-bottom:8px}
}
@media (max-width:480px){
  .fa-hero h1{font-size:26px}.fa-top-meta{display:none}
  .fa-line{grid-template-columns:minmax(48px,72px) minmax(40px,1fr) auto;column-gap:8px}
  .fa-crow{grid-template-columns:minmax(40px,1fr) 44px;column-gap:8px;row-gap:4px;margin:12px 0}
  .fa-crow-label{grid-column:1 / 3;white-space:normal;overflow-wrap:anywhere}
  .fa-axis>span:first-child{display:none}
  .fa-step{grid-template-columns:20px minmax(0,1fr);gap:8px}.fa-idx{width:20px;height:20px;font-size:11px}.fa-steps::before{left:9px}
  .fa-step-meta{margin-left:0;width:100%}
}
@media (prefers-reduced-motion:reduce){
  .fa,.fa *,.fa *::before,.fa *::after{animation-duration:0s !important;animation-iteration-count:1 !important;transition-duration:0s !important}
  .fa-spinner{border-top-color:var(--fa-line-2);border-color:var(--fa-accent)}
}
"""
