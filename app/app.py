"""Frugal Analyst demo. Gradio app and Hugging Face Space entry point.

Run from the repo root:  .venv/bin/python app/app.py   (then open http://127.0.0.1:7860)

It needs no GPU and no model weights. Replay reads trace files, the live tab runs
scripted policies, and a real model option appears only if a backend is available.
Handler logic lives in fa_logic.py. This file only builds the layout and wires events.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for _p in (str(REPO_ROOT), str(APP_DIR)):  # the repo root gives `frugal_analyst`, the app dir gives the fa_* modules
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr  # noqa: E402

import fa_backend  # noqa: E402
import fa_logic as logic  # noqa: E402
from fa_render import CSS as RENDER_CSS  # noqa: E402

TITLE = "Frugal Analyst"
THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    font=("system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"),
)
CSS = RENDER_CSS + """
.gradio-container{max-width:1180px !important;margin:0 auto !important;overflow-x:clip}
html,body{overflow-x:hidden}
.fa-tabs button{font-weight:600}
.fa-table,.fa-table .table-wrap,.fa-table .table-container{min-width:0 !important;max-width:100% !important}
.fa-table{overflow:hidden !important;contain:inline-size}
"""

HERO = """
<div class="fa fa-hero">
  <h1>Frugal Analyst</h1>
  <p>A small model (Tiny Aya) answers questions about a SQLite database with two tools, <code>get_schema</code> and <code>run_sql</code>.
  A right answer is not enough. It must also use few tool calls, pull few rows into its context and cause little database work.</p>
  <p class="fa-tag">This app replays saved runs side by side, runs scripted policies live, lets you try SQL, and shows the results.</p>
</div>
"""

GLOSSARY = """
- **Tool calls.** Each call is one more model turn.
- **Rows returned.** Rows that go back into the model context. An aggregate in SQL returns one row.
- **VM steps.** The SQLite virtual machine instructions a query used. The count is the same on every run, so it is a fair measure of engine work. Time is not.
- **Optimal.** The cost of the efficient scripted path (the oracle) for the same task.
- **Efficiency.** From 0 to 1. The mean of three ratios against optimal: calls, rows and VM steps.
- **Reward.** A correct answer earns 0.5 plus half of its efficiency. A wrong answer earns little. A correct answer needs at least one successful query, so skipping the queries does not pay.
"""


def make_plot(view: dict, metric):
    """Bar chart of one metric as HTML. The axis starts at 0, so equal values still show bars."""
    return logic.chart_html(view, metric)


def make_table(view: dict, full: bool, label: str | None = None):
    """Table with a minimum column width, so headers stay readable and the table scrolls sideways."""
    df = logic.results_table(view, full)
    kwargs = {"label": label} if label else {}
    return gr.Dataframe(df, interactive=False, wrap=False, max_height=420, column_widths=logic.table_widths(df),
                        elem_classes=["fa-table"], **kwargs)


def build_demo(backend: fa_backend.ModelBackend | None = None) -> gr.Blocks:
    backend = backend if backend is not None else fa_backend.default_backend()
    st = logic.initial_compare_state()

    with gr.Blocks(title=TITLE, fill_width=False) as demo:
        gr.HTML(HERO)
        with gr.Accordion("What do the cost numbers mean?", open=False):
            gr.Markdown(GLOSSARY)

        with gr.Tabs(elem_classes="fa-tabs"):
            # ------------------------------------------------------------ Compare
            with gr.Tab("Compare"):
                gr.Markdown("Two models on the same task, step by step. Pick a task or use the buttons.")
                with gr.Row():
                    split = gr.Radio(st["splits"], value=st["split"], label="Split",
                                     info="test_id: known domains. test_ood: a held-out domain.", scale=2, min_width=240)
                    domain = gr.Dropdown(st["domains"], value="all", label="Domain", scale=1, min_width=160)
                    level = gr.Dropdown(logic.LEVEL_CHOICES, value="all", label="Level", scale=1, min_width=140)
                with gr.Row():
                    model_a = gr.Dropdown(st["labels"], value=st["a"], label="Model A", min_width=200)
                    model_b = gr.Dropdown(st["labels"], value=st["b"], label="Model B", min_width=200)
                task = gr.Dropdown(st["tasks"], value=st["task"], label="Task", filterable=True)
                with gr.Row():
                    btn_random = gr.Button("Random task", min_width=160)
                    btn_diff = gr.Button("Show a task where they differ most", variant="primary", min_width=240)
                replay_out = gr.HTML(logic.replay_view(st["split"], st["a"], st["b"], st["task"]))

            # ---------------------------------------------------------------- Ask
            with gr.Tab("Ask"):
                gr.Markdown("Run a policy on a question and watch each step appear.")
                examples0 = logic.example_choices("retail")
                with gr.Row():
                    ask_db = gr.Dropdown(logic.DOMAINS, value="retail", label="Database", min_width=200)
                    ask_example = gr.Dropdown(examples0, value=examples0[0][1] if examples0 else None,
                                              label="Example question", filterable=True, scale=3, min_width=260)
                ask_schema = gr.Code(logic.schema_for("retail"), label="Schema (tables and row counts)", interactive=False,
                                     lines=4, max_lines=8, wrap_lines=True, show_line_numbers=False)
                ask_question = gr.Textbox(
                    logic.question_for_example("retail", examples0[0][1]) if examples0 else "",
                    label="Question", lines=2, max_length=logic.MAX_QUESTION_CHARS,
                    info="Edit it or type your own. Oracle and naive need an example question exactly as listed.",
                )
                ask_policy = gr.Radio(logic.policy_choices(backend), value="oracle", label="Policy")
                gr.HTML(
                    f'<div class="fa fa-muted">{logic.render.esc(logic.backend_note(backend))}</div>'
                )
                ask_run = gr.Button("Run", variant="primary")
                ask_out = gr.HTML(logic.render.notice_html("Press Run to see the steps.", "info"))

            # --------------------------------------------------------- Playground
            with gr.Tab("SQL playground"):
                gr.Markdown(
                    "Run one read-only SELECT. You see the rows and the engine cost, the same way the agent is charged."
                )
                with gr.Row():
                    pg_db = gr.Dropdown(logic.DOMAINS, value="retail", label="Database", min_width=200)
                    pg_pair = gr.Dropdown(logic.pair_choices(), value="0", label="Example pairs",
                                          info="Loads the cheap query. Two ways to the same answer.", scale=3, min_width=260)
                pg_sql = gr.Code(logic.EXAMPLE_PAIRS[0]["cheap"], language="sql", label="Query", lines=6, max_lines=14,
                                 interactive=True, show_line_numbers=False)
                with gr.Row():
                    pg_run = gr.Button("Run query", variant="primary", min_width=160)
                    pg_both = gr.Button("Compare cheap and costly", min_width=220)
                pg_stats = gr.HTML(logic.render.notice_html("Run a query to see its cost.", "info"))
                pg_table = gr.Dataframe(logic._empty_table("run a query"), interactive=False, wrap=True, max_height=380,
                                        label="Result", elem_classes=["fa-table"])
                pg_pair_out = gr.HTML("")

            # ------------------------------------------------------------ Results
            with gr.Tab("Results") as res_tab:
                rv = logic.results_view("all")
                res_split = gr.Radio(logic.results_split_choices(), value="all", label="Split")
                res_note = gr.HTML(logic.render.notice_html(rv["note"], "info"))
                res_head = gr.HTML(rv["headline"])
                metric0 = logic.default_metric(rv)
                res_metric = gr.Dropdown(rv["metrics"], value=metric0, label="Chart metric",
                                         info="Accuracy is in the cards above. Efficiency and tool calls show the cost.")
                res_plot = gr.HTML(logic.chart_html(rv, metric0))
                res_table = make_table(rv, False, label="Key metrics")
                with gr.Accordion("All metrics", open=False):
                    res_table_full = make_table(rv, True, label="All metrics")

        # ================================================================ events
        def cb_view(sp, a, b, t):
            return logic.replay_view(sp, a, b, t)

        def cb_split(sp):
            s = logic.on_split_change(sp)
            return (
                gr.Dropdown(choices=s["labels"], value=s["a"]),
                gr.Dropdown(choices=s["labels"], value=s["b"]),
                gr.Dropdown(choices=s["domains"], value="all"),
                gr.Dropdown(value="all"),
                gr.Dropdown(choices=s["tasks"], value=s["task"]),
            )

        def cb_filter(sp, a, b, dom, lv, cur):
            choices, pick = logic.on_filter_change(sp, a, b, dom, lv, cur)
            return gr.Dropdown(choices=choices, value=pick)

        def cb_random(sp, a, b, dom, lv):
            pick = logic.random_task(sp, a, b, dom, lv)
            return gr.Dropdown(value=pick) if pick else gr.skip()

        def cb_diff(sp, a, b, dom, lv):
            pick = logic.most_different_task(sp, a, b, dom, lv)
            return gr.Dropdown(value=pick) if pick else gr.skip()

        cmp_inputs = [split, model_a, model_b, task]
        filter_inputs = [split, model_a, model_b, domain, level, task]
        split.input(cb_split, split, [model_a, model_b, domain, level, task], api_name="compare_split", concurrency_limit=4
                    ).then(cb_view, cmp_inputs, replay_out, api_name=False)
        for ctl in (domain, level, model_a, model_b):
            ctl.input(cb_filter, filter_inputs, task, api_name=False, concurrency_limit=4
                      ).then(cb_view, cmp_inputs, replay_out, api_name=False)
        task.input(cb_view, cmp_inputs, replay_out, api_name="replay", concurrency_limit=4)
        btn_random.click(cb_random, [split, model_a, model_b, domain, level], task, api_name="random_task", concurrency_limit=4
                         ).then(cb_view, cmp_inputs, replay_out, api_name=False)
        btn_diff.click(cb_diff, [split, model_a, model_b, domain, level], task, api_name="differ_most", concurrency_limit=4
                       ).then(cb_view, cmp_inputs, replay_out, api_name=False)

        def cb_ask_db(db):
            ex = logic.example_choices(db)
            first = ex[0][1] if ex else None
            return logic.schema_for(db), gr.Dropdown(choices=ex, value=first), logic.question_for_example(db, first)

        ask_db.input(cb_ask_db, ask_db, [ask_schema, ask_example, ask_question], api_name=False, concurrency_limit=4)
        ask_example.input(logic.question_for_example, [ask_db, ask_example], ask_question, api_name=False, concurrency_limit=4)

        def cb_ask_run(db, q, pol):
            yield from logic.ask_run(db, q, pol, backend)

        ask_run.click(cb_ask_run, [ask_db, ask_question, ask_policy], ask_out, api_name="ask_run",
                      concurrency_limit=2, time_limit=180, show_progress="hidden")

        pg_pair.input(logic.load_pair, pg_pair, [pg_db, pg_sql], api_name=False, concurrency_limit=4)
        pg_run.click(logic.playground_run, [pg_db, pg_sql], [pg_stats, pg_table], api_name="playground_run", concurrency_limit=3)
        pg_both.click(logic.run_pair, pg_pair, pg_pair_out, api_name="playground_pair", concurrency_limit=2)

        def cb_results(sp, metric):
            v = logic.results_view(sp)
            m = metric if metric in v["metrics"] else logic.default_metric(v)
            return (
                logic.render.notice_html(v["note"], "info"),
                v["headline"],
                gr.Dropdown(choices=v["metrics"], value=m),
                make_plot(v, m),
                make_table(v, False),
                make_table(v, True),
            )

        res_split.input(cb_results, [res_split, res_metric],
                        [res_note, res_head, res_metric, res_plot, res_table, res_table_full],
                        api_name="results", concurrency_limit=4)

        # Fill the tables again when the tab opens. A table that first renders inside a hidden tab can paint empty.
        res_tab.select(cb_results, [res_split, res_metric],
                       [res_note, res_head, res_metric, res_plot, res_table, res_table_full],
                       api_name=False, concurrency_limit=4)

        def cb_metric(sp, metric):
            return make_plot(logic.results_view(sp), metric)

        res_metric.input(cb_metric, [res_split, res_metric], res_plot, api_name=False, concurrency_limit=4)

    # A small queue keeps a public Space from being swamped.
    demo.queue(max_size=24, default_concurrency_limit=4)
    return demo


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Frugal Analyst demo")
    p.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    args = p.parse_args(argv)
    logic.init()
    demo = build_demo()
    demo.launch(
        server_name=args.host, server_port=args.port, theme=THEME, css=CSS,
        max_threads=16, ssr_mode=False, show_error=False, share=False, footer_links=["gradio"],
    )


if __name__ == "__main__":
    main()
