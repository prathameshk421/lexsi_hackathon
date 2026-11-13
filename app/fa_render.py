"""HTML builders. Every piece of text from a trace, a model or a database goes through `esc`.

Only numbers and fixed class names are written into HTML without escaping.
"""

from __future__ import annotations

import html
import re

try:
    from fa_data import LEVEL_TEXT, SCRIPTED_LABELS
except ImportError:  # pragma: no cover
    from .fa_data import LEVEL_TEXT, SCRIPTED_LABELS

MAX_SQL_SHOWN = 1500
MAX_OUTPUT_LINES = 6
MAX_OUTPUT_CHARS = 520
MAX_ANSWER_SHOWN = 300
MAX_QUESTION_SHOWN = 400


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def clip(text, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def num(n) -> str:
    return f"{int(n):,}"


def flt(x, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}"


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


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------


def notice_html(message: str, kind: str = "info", title: str | None = None) -> str:
    head = f'<div class="fa-notice-title">{esc(title)}</div>' if title else ""
    return f'<div class="fa fa-notice fa-notice-{esc(kind)}">{head}<div>{esc(message)}</div></div>'


# ---------------------------------------------------------------------------
# One step, one side
# ---------------------------------------------------------------------------


def step_html(index: int, step: dict) -> str:
    if step["type"] != "tool_call":
        return ""
    ok = step["ok"]
    pills = [f'<span class="fa-pill">{num(step["n_rows"])} rows</span>']
    if step["name"] == "run_sql":
        pills.append(f'<span class="fa-pill">{num(step["vm_steps"])} VM steps</span>')
    pills.append('<span class="fa-pill fa-pill-ok">ok</span>' if ok else '<span class="fa-pill fa-pill-bad">failed</span>')
    args = step["arguments"]
    if step["name"] == "run_sql" and "query" in args:
        body = f'<pre class="fa-code"><code>{esc(format_sql(args.get("query")))}</code></pre>'
        db = args.get("database")
        target = f'<span class="fa-db">{esc(clip(db, 40))}</span>' if db is not None else ""
    else:
        body = ""
        db = args.get("database")
        target = f'<span class="fa-db">{esc(clip(db, 40))}</span>' if db is not None else ""
    preview, hidden = preview_output(step["output"])
    more = f'<div class="fa-more">{hidden} more lines not shown here</div>' if hidden else ""
    return (
        f'<div class="fa-step {"" if ok else "fa-step-bad"}">'
        f'<div class="fa-step-head"><span class="fa-idx">{index}</span>'
        f'<span class="fa-tool">{esc(clip(step["name"], 40))}</span>{target}'
        f'<span class="fa-pills">{"".join(pills)}</span></div>'
        f"{body}"
        f'<pre class="fa-out"><code>{esc(preview)}</code></pre>{more}</div>'
    )


def _verdict(trace: dict, has_gold: bool) -> str:
    if not has_gold:
        return '<span class="fa-badge fa-badge-muted">No gold answer, so no verdict</span>'
    if trace["correct"]:
        return '<span class="fa-badge fa-badge-good">Correct</span>'
    if not trace["answered"]:
        return '<span class="fa-badge fa-badge-bad">No answer</span>'
    if not trace["grounded"]:
        return '<span class="fa-badge fa-badge-bad">Wrong: no successful query</span>'
    return '<span class="fa-badge fa-badge-bad">Wrong</span>'


def final_html(trace: dict, has_gold: bool) -> str:
    finals = [s for s in trace["steps"] if s["type"] == "final"]
    if finals:
        text = clip(finals[-1]["text"], MAX_ANSWER_SHOWN)
        said = f'<pre class="fa-final-text"><code>{esc(text)}</code></pre>'
    else:
        said = '<div class="fa-muted">The run ended without a final answer.</div>'
    extra = ""
    if has_gold and not trace["correct"]:
        extra = f'<div class="fa-muted">Expected: <b>{esc(clip(trace["gold_answer"], 80))}</b></div>'
    return f'<div class="fa-final"><div class="fa-final-head"><span>Final answer</span>{_verdict(trace, has_gold)}</div>{said}{extra}</div>'


def summary_html(trace: dict, has_gold: bool = True) -> str:
    cost, opt = trace["cost"], trace["optimal"]

    def cell(name: str, value: str, ref: int | None) -> str:
        hint = f'<div class="fa-cell-ref">optimal {num(ref)}</div>' if (has_gold and ref is not None) else ""
        return f'<div class="fa-cell"><div class="fa-cell-name">{name}</div><div class="fa-cell-val">{value}</div>{hint}</div>'

    cells = [
        cell("Tool calls", num(cost["tool_calls"]), opt["tool_calls"]),
        cell("Rows returned", num(cost["rows_returned"]), opt["rows"]),
        cell("VM steps", num(cost["vm_steps"]), opt["vm_steps"]),
    ]
    if has_gold:
        cells += [
            cell("Efficiency", flt(trace["efficiency"]), None),
            cell("Reward", flt(trace["reward"]), None),
        ]
    else:
        cells.append(cell("Failed calls", num(cost["failed_calls"]), None))
    return f'<div class="fa-summary">{"".join(cells)}</div>'


def trajectory_html(trace: dict, label: str, side: str = "a", upto: int | None = None,
                    has_gold: bool = True, show_end: bool = True) -> str:
    """One run as a card. `upto` shows only the first N tool steps (used while a live run streams)."""
    calls = [s for s in trace["steps"] if s["type"] == "tool_call"]
    shown = calls if upto is None else calls[:upto]
    steps = "".join(step_html(i, s) for i, s in enumerate(shown, 1))
    if not calls:
        steps = '<div class="fa-muted fa-pad">No tool calls in this run.</div>'
    scripted = '<span class="fa-pill">scripted reference</span>' if label in SCRIPTED_LABELS else ""
    end = final_html(trace, has_gold) + summary_html(trace, has_gold) if show_end else ""
    return (
        f'<section class="fa-card fa-side-{esc(side)}">'
        f'<header class="fa-card-head"><span class="fa-dot"></span><span class="fa-model">{esc(clip(label, 40))}</span>{scripted}</header>'
        f"{steps}{end}</section>"
    )


# ---------------------------------------------------------------------------
# Cost meters
# ---------------------------------------------------------------------------


def _ratio_text(value: int, ref: int | None) -> str:
    if not ref:
        return ""
    r = value / ref
    return "at optimal" if abs(r - 1) < 0.05 else f"{r:,.1f}x optimal"


def _bar_class(value: int, ref: int | None) -> str:
    """Neutral bar at or near the optimal cost, warm hatched bar above it."""
    return "fa-bar-ok" if ref is None or value <= ref * 1.05 else "fa-bar-warn"


def meters_html(ta: dict, la: str, tb: dict, lb: str) -> str:
    rows = []
    spec = [
        ("Tool calls", "tool_calls", "tool_calls"),
        ("Rows returned", "rows_returned", "rows"),
        ("VM steps", "vm_steps", "vm_steps"),
    ]
    for title, ck, ok_ in spec:
        va, vb = ta["cost"][ck], tb["cost"][ck]
        ref = ta["optimal"].get(ok_)
        scale = max(va, vb, ref or 0, 1)
        lines = []
        for side, label, v in (("a", la, va), ("b", lb, vb)):
            width = max(v / scale * 100, 1.5 if v else 0)
            lines.append(
                f'<div class="fa-line"><span class="fa-line-label">{esc(clip(label, 24))}</span>'
                f'<div class="fa-track"><div class="fa-bar {_bar_class(v, ref)}" style="width:{width:.1f}%"></div>'
                + (f'<span class="fa-tick" style="left:{ref / scale * 100:.1f}%"></span>' if ref else "")
                + f'</div><span class="fa-line-val">{num(v)}</span>'
                f'<span class="fa-line-ratio">{esc(_ratio_text(v, ref))}</span></div>'
            )
        ref_txt = f"optimal {num(ref)}" if ref is not None else ""
        rows.append(
            f'<div class="fa-meter"><div class="fa-meter-title"><span>{title}</span>'
            f'<span class="fa-muted">{ref_txt}</span></div>{"".join(lines)}</div>'
        )
    return (
        '<div class="fa-panel"><div class="fa-panel-title">Cost at a glance <span class="fa-muted">'
        "(shorter bar is cheaper, the tick marks the optimal cost)</span></div>"
        f'<div class="fa-meters">{"".join(rows)}</div></div>'
    )


# ---------------------------------------------------------------------------
# Replay and live pages
# ---------------------------------------------------------------------------


def question_html(trace: dict, show_gold: bool = True) -> str:
    level = trace["level"]
    meta = [trace["task_id"]]
    if trace["domain"]:
        meta.append(trace["domain"])
    if level:
        meta.append(f"level {level}")
    note = f'<div class="fa-muted">{esc(LEVEL_TEXT[level])}</div>' if level in LEVEL_TEXT else ""
    gold = (
        f'<div class="fa-muted">Gold answer: <b>{esc(clip(trace["gold_answer"], 80))}</b></div>' if show_gold else ""
    )
    return (
        f'<div class="fa-question"><div class="fa-meta">{esc(" / ".join(meta))}</div>'
        f'<div class="fa-q">{esc(clip(trace["question"], MAX_QUESTION_SHOWN))}</div>{note}{gold}</div>'
    )


def replay_html(ta: dict, la: str, tb: dict, lb: str) -> str:
    return (
        '<div class="fa">'
        f"{question_html(ta)}{meters_html(ta, la, tb, lb)}"
        f'<div class="fa-cols">{trajectory_html(ta, la, "a")}{trajectory_html(tb, lb, "b")}</div></div>'
    )


def live_html(trace: dict, label: str, upto: int, done: bool, has_gold: bool, question: str, note: str = "") -> str:
    head = (
        question_html(trace, show_gold=has_gold)
        if has_gold
        else f'<div class="fa-question"><div class="fa-meta">typed question</div><div class="fa-q">{esc(clip(question, MAX_QUESTION_SHOWN))}</div></div>'
    )
    note_html = f'<div class="fa-muted fa-pad">{esc(note)}</div>' if note else ""
    status = "" if done else '<div class="fa-muted fa-pad">Running...</div>'
    return (
        f'<div class="fa">{head}{note_html}'
        f'{trajectory_html(trace, label, "a", upto=upto, has_gold=has_gold, show_end=done)}{status}</div>'
    )


# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------


def query_stats_html(n_rows: int, vm_steps: int, capped: bool, hint: str, error: str | None = None, note: str = "") -> str:
    if error:
        return (
            '<div class="fa fa-notice fa-notice-bad"><div class="fa-notice-title">Query refused or failed</div>'
            f"<div>{esc(clip(error, 400))}</div>"
            f'<div class="fa-muted">Cost so far: {num(vm_steps)} VM steps</div></div>'
        )
    rows = f"{num(n_rows)}{'+' if capped else ''}"
    return (
        '<div class="fa"><div class="fa-summary">'
        f'<div class="fa-cell"><div class="fa-cell-name">Rows returned</div><div class="fa-cell-val">{rows}</div></div>'
        f'<div class="fa-cell"><div class="fa-cell-name">VM steps</div><div class="fa-cell-val">{num(vm_steps)}</div></div>'
        f"</div>"
        + (f'<div class="fa-hint">{esc(hint)}</div>' if hint else "")
        + (f'<div class="fa-muted fa-pad">{esc(note)}</div>' if note else "")
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
            stats = f'<div class="fa-notice fa-notice-bad">{esc(clip(err, 300))}</div>'
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
            f'<section class="fa-card fa-pair-{"cheap" if n == 0 else "costly"}"><header class="fa-card-head"><span class="fa-dot"></span>'
            f'<span class="fa-model">{esc(title)}</span></header>'
            f'<pre class="fa-code"><code>{esc(format_sql(sql))}</code></pre>{stats}</section>'
        )
    return f'<div class="fa"><div class="fa-cols">{"".join(cards)}</div></div>'


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


def headline_html(before_label: str, after_label: str, cards: list[tuple[str, float, float]], source: str) -> str:
    if not cards:
        return ""
    out = []
    for card in cards:
        name, b, a = card[:3]
        extra = card[3] if len(card) > 3 else ""
        delta = a - b
        sign = "+" if delta > 0 else ""
        extra_html = f'<div class="fa-cell-ref">{esc(extra)}</div>' if extra else ""
        out.append(
            f'<div class="fa-stat"><div class="fa-cell-name">{esc(pretty_key(name))}</div>'
            f'<div class="fa-stat-vals"><span class="fa-stat-before">{esc(_fmt_value(b))}</span>'
            f'<span class="fa-arrow" aria-label="to">to</span>'
            f'<span class="fa-stat-after">{esc(_fmt_value(a))}</span></div>'
            f'<div class="fa-cell-ref">change {esc(sign + _fmt_value(round(delta, 4)))}</div>{extra_html}</div>'
        )
    return (
        f'<div class="fa"><div class="fa-panel-title">{esc(before_label)} to {esc(after_label)} '
        f'<span class="fa-muted">({esc(source)})</span></div><div class="fa-stats">{"".join(out)}</div></div>'
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
            f'<div class="fa-crow"><span class="fa-crow-label">{esc(clip(name, 40))}</span>'
            f'<div class="fa-track"><div class="fa-bar {cls}" style="width:{width:.1f}%"></div></div>'
            f'<span class="fa-line-val">{esc(_tick(value))}</span></div>'
        )
    axis = (
        '<div class="fa-crow fa-axis"><span></span><div class="fa-axis-ticks">'
        f"<span>0</span><span>{_tick(upper / 2)}</span><span>{_tick(upper)}</span></div><span></span></div>"
    )
    return (
        f'<div class="fa fa-panel"><div class="fa-panel-title">{esc(metric)}</div>'
        f'<div class="fa-legend">{legend}<span class="fa-legend-item"><span class="fa-swatch fa-hatch fa-g-none"></span>'
        f"hatched: scripted reference policy</span></div>{''.join(rows)}{axis}</div>"
    )


# ---------------------------------------------------------------------------
# CSS. Colours use Gradio theme variables, so light and dark both work.
# ---------------------------------------------------------------------------

CSS = """
.fa{--fa-a:#4f46e5;--fa-b:#0f766e;--fa-ok:#64748b;--fa-over:#c2410c;--fa-good:#15803d;--fa-bad:#b91c1c;--fa-warn:#b45309;
  --fa-muted:var(--body-text-color-subdued);--fa-line:var(--border-color-primary);
  --fa-soft:var(--background-fill-secondary);--fa-card:var(--block-background-fill);
  color:var(--body-text-color);font-size:15px;line-height:1.45;max-width:100%}
.dark .fa{--fa-a:#a5b4fc;--fa-b:#5eead4;--fa-ok:#94a3b8;--fa-over:#fb923c;--fa-good:#4ade80;--fa-bad:#f87171;--fa-warn:#fbbf24}
.fa *{box-sizing:border-box}
.fa-muted{color:var(--fa-muted);font-size:13px}
.fa-pad{padding:8px 4px}
.fa-question{border-left:4px solid var(--fa-ok);padding:6px 14px;margin:0 0 14px;background:var(--fa-soft);border-radius:0 8px 8px 0}
.fa-meta{font-size:12px;color:var(--fa-muted);text-transform:uppercase;letter-spacing:.04em}
.fa-q{font-size:18px;font-weight:600;margin:2px 0 4px;overflow-wrap:anywhere}
.fa-panel{border:1px solid var(--fa-line);border-radius:12px;padding:14px 16px;margin:0 0 14px;background:var(--fa-card)}
.fa-panel-title{font-weight:600;margin-bottom:10px}
.fa-meters{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px}
.fa-meter-title{display:flex;justify-content:space-between;gap:8px;font-weight:600;font-size:14px;margin-bottom:4px}
.fa-line{display:grid;grid-template-columns:78px minmax(60px,1fr) auto;column-gap:8px;align-items:center;margin:3px 0}
.fa-line-label{font-size:12px;color:var(--fa-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fa-track{position:relative;height:12px;background:var(--fa-soft);border-radius:6px}
.fa-bar{height:100%;border-radius:6px;min-width:0}
.fa-bar-ok{background:var(--fa-ok)}
.fa-bar-warn{background:repeating-linear-gradient(135deg,var(--fa-over) 0 6px,transparent 6px 9px),var(--fa-over);
  box-shadow:inset 0 0 0 1px var(--fa-over)}
.fa-tick{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--body-text-color);opacity:.7}
.fa-line-val{font-variant-numeric:tabular-nums;font-weight:600;font-size:13px;text-align:right}
.fa-line-ratio{grid-column:2 / 4;font-size:11px;color:var(--fa-muted);text-align:right;margin-top:-2px}
.fa-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:14px;align-items:start}
.fa-card{border:1px solid var(--fa-line);border-radius:12px;background:var(--fa-card);padding:12px;min-width:0}
.fa-card-head{display:flex;align-items:center;gap:8px;font-weight:700;margin-bottom:10px}
.fa-dot{width:10px;height:10px;border-radius:50%;background:var(--fa-ok)}
.fa-pair-costly .fa-dot{background:var(--fa-over)}
.fa-step{border:1px solid var(--fa-line);border-left:3px solid var(--fa-good);border-radius:8px;padding:8px 10px;margin:0 0 8px;min-width:0}
.fa-step-bad{border-left-color:var(--fa-bad)}
.fa-step-head{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:6px}
.fa-idx{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:50%;background:var(--fa-soft);font-size:12px;font-weight:700}
.fa-tool{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:600;font-size:13px}
.fa-db{font-size:12px;color:var(--fa-muted)}
.fa-pills{display:flex;flex-wrap:wrap;gap:4px;margin-left:auto}
.fa-pill{font-size:11px;padding:1px 7px;border:1px solid var(--fa-line);border-radius:999px;background:var(--fa-soft);white-space:nowrap}
.fa-pill-ok{color:var(--fa-good)}.fa-pill-bad{color:var(--fa-bad);border-color:var(--fa-bad);font-weight:700}
.fa pre{display:block;animation:none !important;transition:none !important;content-visibility:visible !important;contain:none !important;opacity:1 !important;margin:4px 0 0;padding:8px 10px;border-radius:6px;overflow-x:auto;max-width:100%;
  font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.4;white-space:pre;
  background:var(--fa-soft);color:var(--body-text-color);border:none}
.fa pre code{background:none;padding:0;font:inherit;color:inherit}
.fa-out{max-height:170px;overflow:auto}
.fa-code{border-left:3px solid var(--fa-ok)}
.fa-more{font-size:11px;color:var(--fa-muted);margin-top:2px}
.fa-final{margin-top:10px;padding:10px;border:1px solid var(--fa-line);border-radius:8px;background:var(--fa-soft)}
.fa-final-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;font-weight:600}
.fa-final-text{background:transparent !important;padding:4px 0 !important;white-space:pre-wrap !important;overflow-wrap:anywhere;font-size:14px !important}
.fa-badge{font-size:12px;font-weight:700;padding:2px 10px;border-radius:999px;border:1px solid currentColor}
.fa-badge-good{color:var(--fa-good)}.fa-badge-bad{color:var(--fa-bad)}.fa-badge-muted{color:var(--fa-muted);font-weight:500}
.fa-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:8px;margin-top:10px}
.fa-cell{border:1px solid var(--fa-line);border-radius:8px;padding:6px 10px;background:var(--fa-card)}
.fa-cell-name{font-size:12px;color:var(--fa-muted)}
.fa-cell-val{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}
.fa-cell-ref{font-size:11px;color:var(--fa-muted)}
.fa-notice{border:1px solid var(--fa-line);border-radius:10px;padding:12px 14px;background:var(--fa-soft)}
.fa-notice-title{font-weight:700;margin-bottom:2px}
.fa-notice-bad{border-color:var(--fa-bad)}.fa-notice-warn{border-color:var(--fa-warn)}
.fa-hint{margin-top:8px;padding:8px 12px;border-radius:8px;background:var(--fa-soft);border-left:3px solid var(--fa-warn);font-size:14px}
.fa-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.fa-stat{border:1px solid var(--fa-line);border-radius:10px;padding:10px 12px;background:var(--fa-card)}
.fa-stat-vals{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.fa-stat-before{font-size:16px;color:var(--fa-muted);font-variant-numeric:tabular-nums}
.fa-stat-after{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.fa-arrow{font-size:12px;color:var(--fa-muted)}
.fa-crow{display:grid;grid-template-columns:minmax(80px,min(200px,34%)) minmax(60px,1fr) 44px;column-gap:8px;align-items:center;margin:5px 0}
.fa-crow-label{font-size:13px;overflow-wrap:anywhere}
.fa-crow .fa-track{height:16px}
.fa-axis{margin-top:2px}
.fa-axis-ticks{display:flex;justify-content:space-between;font-size:11px;color:var(--fa-muted);border-top:1px solid var(--fa-line);padding-top:2px}
.fa-legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-bottom:8px;font-size:12px;color:var(--fa-muted)}
.fa-legend-item{display:inline-flex;align-items:center;gap:6px}
.fa-swatch{width:12px;height:12px;border-radius:3px;display:inline-block}
.fa-g0{background-color:#4f6fa8}.fa-g1{background-color:#8a5fb0}.fa-g2{background-color:#2f8f80}.fa-g3{background-color:#a58a2a}
.dark .fa-g0{background-color:#8fb0e8}.dark .fa-g1{background-color:#c2a0ea}.dark .fa-g2{background-color:#6fd0c0}.dark .fa-g3{background-color:#e0c76a}
.fa-g-none{background-color:var(--fa-soft);border:1px solid var(--fa-line)}
.fa-hatch{background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.45) 0 3px,transparent 3px 7px)}
.dark .fa-hatch{background-image:repeating-linear-gradient(135deg,rgba(0,0,0,.4) 0 3px,transparent 3px 7px)}
.fa-hero{padding:4px 2px 8px}
.fa-hero h1{font-size:26px;margin:0 0 4px;font-weight:800}
.fa-hero p{margin:0 0 6px;font-size:15px;max-width:78ch}
.fa-hero .fa-tag{color:var(--fa-muted)}
@media (max-width:520px){.fa-q{font-size:16px}.fa-line{grid-template-columns:60px minmax(40px,1fr) auto}.fa-hero h1{font-size:22px}}
"""
