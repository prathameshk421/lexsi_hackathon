"""Frugal Analyst demo. Gradio app and Hugging Face Space entry point.

Run from the repo root:  .venv/bin/python app/app.py   (then open http://127.0.0.1:7860)

It needs no GPU and no model weights. Replay reads trace files, the live tab runs
scripted policies, and a real model option appears only if a backend is available.
Handler logic lives in fa_logic.py, HTML in fa_render.py. This file builds the layout,
the theme and the event wiring. Design notes are in DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
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

render = logic.render

TITLE = "Frugal Analyst"
EVENT = "Lexsi Labs hackathon, Agentic Fine-tuning track"

# Footer links. Each one is read from an environment variable, else from links.json next to this file
# (space/build_space.py writes it for a Space). A link is hidden when unset or not a plain https URL.
LINK_ENV = {
    "GitHub repository": "FRUGAL_LINK_GITHUB",
    "Model collection": "FRUGAL_LINK_MODELS",
    "Dataset collection": "FRUGAL_LINK_DATASETS",
    "Write-up": "FRUGAL_LINK_WRITEUP",
}
LINKS_FILE = APP_DIR / "links.json"


def read_links_file(path: Path | None = None) -> dict[str, str]:
    """Variable name -> URL from a small JSON file. A missing or bad file gives no links."""
    p = Path(path) if path is not None else LINKS_FILE
    try:
        if not p.is_file() or p.stat().st_size > 8192:
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(obj, dict):
        return {}
    return {k: v for k, v in obj.items() if k in LINK_ENV.values() and isinstance(v, str)}


def footer_links(env=None, links_file: Path | None = None) -> list[tuple[str, str]]:
    env = os.environ if env is None else env
    baked = read_links_file(links_file)
    return [(text, env.get(var) or baked.get(var, "")) for text, var in LINK_ENV.items()]


def base_model_name(env=None) -> str:
    env = os.environ if env is None else env
    return env.get("FRUGAL_MODEL_ID") or fa_backend.DEFAULT_MODEL_ID


# ---------------------------------------------------------------------------
# Theme. Fonts load from one Google Fonts stylesheet. The stacks fall back to system fonts,
# so the page keeps its layout when the font cannot load.
# ---------------------------------------------------------------------------

FONT_SANS = ("IBM Plex Sans", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue",
             "Arial", "sans-serif")
FONT_MONO = ("IBM Plex Mono", "ui-monospace", "SF Mono", "Menlo", "Consolas", "Liberation Mono", "monospace")
HEAD = (
    '<meta name="color-scheme" content="light dark">'
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
    '&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
)

INK, INK_D = "#16181d", "#ecedee"
SUB, SUB_D = "#575b63", "#a3a7ae"
LINE, LINE_D = "#e4e4e0", "#2a2c31"
LINE2, LINE2_D = "#d2d2cd", "#3a3d44"
SURF, SURF_D = "#ffffff", "#131417"
SUNK, SUNK_D = "#f4f4f2", "#1a1c20"
BG, BG_D = "#fafaf9", "#0b0c0e"
ACC, ACC_D = "#2b59d9", "#87a5ff"
ACC_SOFT, ACC_SOFT_D = "#ebf0fd", "rgba(135,165,255,.14)"

THEME = gr.themes.Base(font=FONT_SANS, font_mono=FONT_MONO).set(
    body_background_fill=BG, body_background_fill_dark=BG_D,
    body_text_color=INK, body_text_color_dark=INK_D,
    body_text_color_subdued=SUB, body_text_color_subdued_dark=SUB_D,
    body_text_size="15px",
    background_fill_primary=SURF, background_fill_primary_dark=SURF_D,
    background_fill_secondary=SUNK, background_fill_secondary_dark=SUNK_D,
    border_color_primary=LINE, border_color_primary_dark=LINE_D,
    border_color_accent=ACC, border_color_accent_dark=ACC_D,
    color_accent=ACC,
    color_accent_soft=ACC_SOFT, color_accent_soft_dark=ACC_SOFT_D,
    link_text_color=ACC, link_text_color_dark=ACC_D,
    link_text_color_hover=ACC, link_text_color_hover_dark=ACC_D,
    link_text_color_visited=ACC, link_text_color_visited_dark=ACC_D,
    code_background_fill=SUNK, code_background_fill_dark=SUNK_D,
    shadow_drop="none", shadow_drop_lg="0 8px 24px rgba(0,0,0,.10)", shadow_spread="3px",
    block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_width="0px", block_shadow="none", block_padding="0px", block_radius="6px",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_border_width="0px", block_label_text_color=SUB, block_label_text_color_dark=SUB_D,
    block_label_text_size="13px", block_label_text_weight="500", block_label_padding="0 0 6px 0",
    block_label_margin="0", block_label_radius="0", block_label_shadow="none",
    block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
    block_title_border_width="0px", block_title_text_color=INK, block_title_text_color_dark=INK_D,
    block_title_text_size="13px", block_title_text_weight="500", block_title_padding="0", block_title_radius="0",
    block_info_text_color=SUB, block_info_text_color_dark=SUB_D, block_info_text_size="12px",
    container_radius="10px", layout_gap="16px", form_gap_width="0px",
    panel_background_fill="transparent", panel_background_fill_dark="transparent", panel_border_width="0px",
    section_header_text_size="14px", section_header_text_weight="600", prose_text_size="15px",
    accordion_text_color=INK, accordion_text_color_dark=INK_D,
    input_background_fill=SURF, input_background_fill_dark=SURF_D,
    input_background_fill_focus=SURF, input_background_fill_focus_dark=SURF_D,
    input_background_fill_hover=SURF, input_background_fill_hover_dark=SURF_D,
    input_border_color=LINE2, input_border_color_dark=LINE2_D,
    input_border_color_hover="#a9aaa5", input_border_color_hover_dark="#55585f",
    input_border_color_focus=ACC, input_border_color_focus_dark=ACC_D,
    input_border_width="1px", input_radius="6px", input_padding="8px 12px", input_text_size="14px",
    input_shadow="none", input_shadow_focus=f"0 0 0 3px {ACC_SOFT}", input_shadow_focus_dark=f"0 0 0 3px {ACC_SOFT_D}",
    input_placeholder_color="#8a8e95", input_placeholder_color_dark="#7c818a",
    checkbox_label_background_fill=SURF, checkbox_label_background_fill_dark=SURF_D,
    checkbox_label_background_fill_hover=SUNK, checkbox_label_background_fill_hover_dark=SUNK_D,
    checkbox_label_background_fill_selected=ACC_SOFT, checkbox_label_background_fill_selected_dark=ACC_SOFT_D,
    checkbox_label_border_color=LINE2, checkbox_label_border_color_dark=LINE2_D,
    checkbox_label_border_color_selected=ACC, checkbox_label_border_color_selected_dark=ACC_D,
    checkbox_label_border_width="1px", checkbox_label_padding="8px 12px", checkbox_label_gap="8px",
    checkbox_label_text_size="14px", checkbox_label_text_weight="400",
    checkbox_label_text_color=INK, checkbox_label_text_color_dark=INK_D,
    checkbox_label_text_color_selected=INK, checkbox_label_text_color_selected_dark=INK_D,
    checkbox_label_shadow="none",
    checkbox_background_color=SURF, checkbox_background_color_dark=SURF_D,
    checkbox_background_color_selected=ACC, checkbox_background_color_selected_dark=ACC_D,
    checkbox_border_color=LINE2, checkbox_border_color_dark=LINE2_D,
    checkbox_border_color_selected=ACC, checkbox_border_color_selected_dark=ACC_D,
    checkbox_border_color_focus=ACC, checkbox_border_color_focus_dark=ACC_D,
    loader_color=ACC, loader_color_dark=ACC_D, slider_color=ACC, slider_color_dark=ACC_D,
    button_border_width="1px", button_transform_hover="none", button_transform_active="none",
    button_transition="background-color .12s ease, border-color .12s ease, color .12s ease",
    button_large_padding="0 16px", button_large_radius="6px", button_large_text_size="14px", button_large_text_weight="500",
    button_medium_padding="0 16px", button_medium_radius="6px", button_medium_text_size="14px",
    button_medium_text_weight="500",
    button_small_padding="0 12px", button_small_radius="6px", button_small_text_size="13px", button_small_text_weight="500",
    button_primary_background_fill=INK, button_primary_background_fill_dark=INK_D,
    button_primary_background_fill_hover="#2c3038", button_primary_background_fill_hover_dark="#ffffff",
    button_primary_border_color=INK, button_primary_border_color_dark=INK_D,
    button_primary_border_color_hover="#2c3038", button_primary_border_color_hover_dark="#ffffff",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#0b0c0e",
    button_primary_text_color_hover="#ffffff", button_primary_text_color_hover_dark="#0b0c0e",
    button_primary_shadow="none", button_primary_shadow_hover="none", button_primary_shadow_active="none",
    button_secondary_background_fill=SURF, button_secondary_background_fill_dark=SURF_D,
    button_secondary_background_fill_hover=SUNK, button_secondary_background_fill_hover_dark=SUNK_D,
    button_secondary_border_color=LINE2, button_secondary_border_color_dark=LINE2_D,
    button_secondary_border_color_hover="#a9aaa5", button_secondary_border_color_hover_dark="#55585f",
    button_secondary_text_color=INK, button_secondary_text_color_dark=INK_D,
    button_secondary_text_color_hover=INK, button_secondary_text_color_hover_dark=INK_D,
    button_secondary_shadow="none", button_secondary_shadow_hover="none", button_secondary_shadow_active="none",
    error_background_fill=SURF, error_background_fill_dark=SURF_D,
    error_border_color="#c62828", error_border_color_dark="#f87171",
    error_text_color="#c62828", error_text_color_dark="#f87171",
)

# Page shell. These rules restyle Gradio's own markup (tabs, rows, toolbars) and hide its footer.
SHELL_CSS = """
html,body{overflow-x:hidden}
body{background:var(--fa-bg) !important;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility}
.gradio-container{max-width:1200px !important;width:100% !important;margin:0 auto !important;padding:0 32px !important;
  overflow-x:clip;background:transparent !important;font-family:var(--fa-sans)}
.gradio-container .main,.gradio-container .wrap,.gradio-container .contain{max-width:100% !important}
html body .gradio-container{padding:0 32px !important}
html body .gradio-container>.main{padding-left:0 !important;padding-right:0 !important}
.gradio-container>.main>.wrap{padding:0 !important}
footer[aria-label="Gradio footer navigation"]{display:none !important}
.gradio-container .prose{font-size:15px}
.gradio-container .html-container{padding:0 !important}

/* tabs as a quiet underline nav */
.fa-tabs{gap:0 !important}
.fa-tabs .tab-wrapper{border-bottom:1px solid var(--fa-line);margin-bottom:24px;height:auto !important;padding:0 !important}
.fa-tabs .tab-container{gap:4px;height:auto !important;overflow:visible}
.fa-tabs .tab-container button,.fa-tabs .overflow-menu button{font-family:var(--fa-sans) !important;font-size:14px !important;
  font-weight:500 !important;color:var(--fa-text-2) !important;padding:0 12px !important;height:44px !important;
  border:0 !important;background:none !important;border-radius:0 !important;position:relative;margin:0 !important}
.fa-tabs .tab-container button:first-child{padding-left:0 !important}
.fa-tabs .tab-container button:hover{color:var(--fa-text) !important}
.fa-tabs .tab-container button.selected{color:var(--fa-text) !important}
.fa-tabs .tab-container button::after{display:none !important}
.fa-tabs .tab-container button.selected::before{content:"";position:absolute;left:12px;right:12px;bottom:-1px;height:2px;
  background:var(--fa-text);border-radius:1px}
.fa-tabs .tab-container button.selected:first-child::before{left:0}
.fa-tabs .tabitem{padding:0 !important;border:0 !important;background:transparent !important}

/* toolbars that hold the controls */
.fa-toolbar{background:var(--fa-surface) !important;border:1px solid var(--fa-line) !important;border-radius:10px !important;
  padding:20px !important;gap:16px !important;margin:0 0 24px !important}
.fa-toolbar .row,.fa-toolbar>.form{gap:16px !important}
.gradio-container .form{background:transparent !important;border:0 !important;box-shadow:none !important;gap:16px !important;overflow:visible}
.gradio-container .block{overflow:visible}
.fa-actions{align-items:center !important;gap:8px !important}
.fa-actions button{flex:0 0 auto !important;width:auto !important;min-width:0 !important}
.gradio-container button.lg,.gradio-container button.md,.gradio-container button.sm{min-height:40px}
.gradio-container button:focus-visible,.gradio-container [role="tab"]:focus-visible,.gradio-container summary:focus-visible{
  outline:2px solid var(--fa-accent) !important;outline-offset:2px !important}
.gradio-container button:disabled,.gradio-container button[disabled]{opacity:.55;cursor:progress}
.gradio-container input,.gradio-container textarea{font-family:var(--fa-sans)}
.gradio-container [data-testid="block-info"]{font-weight:500}
.gradio-container .wrap-inner,.gradio-container .secondary-wrap{min-height:24px}
.gradio-container .secondary-wrap input[role="combobox"]{text-overflow:ellipsis;min-width:0;padding-right:8px}
.gradio-container .secondary-wrap .icon-wrap{flex:none}

/* segmented radio */
.fa-seg>[data-testid="block-info"],.fa-options>[data-testid="block-info"]{display:block;margin-bottom:8px}
.fa-seg .wrap{display:flex !important;flex-wrap:nowrap;gap:2px !important;padding:2px;border:1px solid var(--fa-line-2);
  border-radius:6px;background:var(--fa-sunken);width:100% !important;max-width:100%}
.fa-seg .wrap label{flex:1 1 auto;justify-content:center;white-space:nowrap;min-width:0}
.fa-seg .wrap label span{overflow:hidden;text-overflow:ellipsis}
.fa-seg .wrap label{border:0 !important;background:transparent !important;border-radius:4px !important;padding:0 12px !important;
  min-height:36px;box-shadow:none !important;color:var(--fa-text-2) !important;margin:0 !important;position:relative;
  display:inline-flex;align-items:center}
.fa-seg .wrap label:hover{color:var(--fa-text) !important}
.fa-seg .wrap label.selected{background:var(--fa-surface) !important;box-shadow:0 0 0 1px var(--fa-line-2) !important;
  color:var(--fa-text) !important;font-weight:500}
.fa-seg .wrap label input{position:absolute !important;opacity:0 !important;width:1px !important;height:1px !important;
  margin:0 !important;pointer-events:none}
.fa-seg .wrap label span{margin-left:0 !important}
.fa-seg .wrap label:has(input:focus-visible){outline:2px solid var(--fa-accent);outline-offset:1px}

/* stacked option list (policy) */
.fa-options .wrap{flex-direction:column !important;gap:8px !important}
.fa-options .wrap label{width:100%;min-height:44px;border-radius:6px !important}
.fa-options .wrap label:has(input:focus-visible){outline:2px solid var(--fa-accent);outline-offset:1px}

/* two column work area (Ask) */
.fa-split{gap:24px !important;align-items:flex-start !important}
.fa-split>.column{min-width:0 !important}
.fa-out-area{min-height:320px}
.fa-section-title{font-size:13px;font-weight:600;color:var(--fa-text);margin:8px 0 12px}
.fa-code-editor{border:1px solid var(--fa-line-2) !important;border-radius:6px !important;overflow:hidden}
.fa-code-editor .cm-editor{font-family:var(--fa-mono) !important;font-size:13px !important}
body:not(.dark) .fa-code-editor .cm-line span{color:var(--fa-text) !important}  /* the editor's light token colours fail AA */
body:not(.dark) .fa-code-editor .cm-line{color:var(--fa-text)}
.fa-code-editor:focus-within{border-color:var(--fa-accent) !important;box-shadow:0 0 0 3px var(--fa-accent-soft)}
.fa-gap{height:8px}
@media (max-width:767px){
  html body .gradio-container{padding:0 16px !important}
  .fa-toolbar{padding:16px !important}
  .fa-tabs .tab-container{gap:0}
  .fa-tabs .tab-container button{padding:0 8px !important}
  .fa-tabs .tab-container button.selected::before{left:8px;right:8px}
}
@media (prefers-reduced-motion:reduce){.gradio-container *{transition-duration:0s !important;animation-duration:0s !important}}
"""

CSS = RENDER_CSS + SHELL_CSS

HERO = render.header_html(EVENT) + render.hero_html()
FOOTER_TEXT = "Frugal Analyst"


def make_plot(view: dict, metric):
    """Bar chart of one metric as HTML. The axis starts at 0, so equal values still show bars."""
    return logic.chart_html(view, metric)


def make_table(view: dict, full: bool) -> str:
    """The results table as HTML. The full table sits in a native disclosure."""
    table = logic.results_table_html(view, full)
    if not full:
        return table
    if not view["rows"]:
        return ""
    return f'<div class="fa"><details class="fa-more-table"><summary>All metrics</summary><div class="fa-gap"></div>{table}</details></div>'


def build_demo(backend: fa_backend.ModelBackend | None = None) -> gr.Blocks:
    backend = backend if backend is not None else fa_backend.default_backend()
    st = logic.initial_compare_state()

    legend = logic.runs_legend()
    with gr.Blocks(title=TITLE, fill_width=False) as demo:
        gr.HTML(render.header_html(EVENT) + render.hero_html(logic.findings_text()), padding=False)

        with gr.Tabs(elem_classes="fa-tabs"):
            # ------------------------------------------------------------ Compare
            with gr.Tab("Compare"):
                with gr.Column(elem_classes="fa-toolbar"):
                    with gr.Row():
                        model_a = gr.Dropdown(st["labels"], value=st["a"], label="Run A", scale=2, min_width=136)
                        model_b = gr.Dropdown(st["labels"], value=st["b"], label="Run B", scale=2, min_width=136)
                        split = gr.Radio(logic.split_options(st["splits"]), value=st["split"], label="Test set",
                                         scale=3, min_width=260, elem_classes="fa-seg")
                    with gr.Row():
                        task = gr.Dropdown(st["tasks"], value=st["task"], label=logic.task_info(len(st["tasks"])),
                                           filterable=True, scale=5, min_width=260)
                        domain = gr.Dropdown(st["domains"], value="all", label="Domain", scale=1, min_width=136)
                        level = gr.Dropdown(logic.LEVEL_CHOICES, value="all", label="Level", scale=1, min_width=120)
                    with gr.Row(elem_classes="fa-actions"):
                        btn_diff = gr.Button("Largest difference", variant="primary", size="md", scale=0)
                        btn_random = gr.Button("Random task", variant="secondary", size="md", scale=0)
                gr.HTML(legend, padding=False)
                replay_out = gr.HTML(logic.replay_view(st["split"], st["a"], st["b"], st["task"]), padding=False,
                                     elem_classes="fa-out-area")

            # ---------------------------------------------------------------- Ask
            with gr.Tab("Ask"):
                examples0 = logic.example_choices("retail")
                live = backend is not None and backend.available()
                if not live:  # said once, at the top, so nobody waits for a model that is not there
                    gr.HTML(render.notice_html(logic.backend_note(backend), "info", "Live model inference is off"),
                            padding=False)
                    gr.HTML('<div class="fa-gap"></div>', padding=False)
                with gr.Row(elem_classes="fa-split", equal_height=False):
                    with gr.Column(scale=4, min_width=300, elem_classes="fa-toolbar"):
                        ask_db = gr.Dropdown(logic.DOMAINS, value="retail", label="Database")
                        ask_example = gr.Dropdown(examples0, value=examples0[0][1] if examples0 else None,
                                                  label="Example question", filterable=True)
                        ask_question = gr.Textbox(
                            logic.question_for_example("retail", examples0[0][1]) if examples0 else "",
                            label="Question", lines=3, max_length=logic.MAX_QUESTION_CHARS,
                            info=f"Up to {logic.MAX_QUESTION_CHARS} characters. Oracle and naive answer only the example questions.",
                        )
                        ask_policy = gr.Radio(logic.policy_choices(backend), value="oracle", label="Policy",
                                              elem_classes="fa-options")
                        if live:
                            gr.HTML(f'<div class="fa fa-muted">{render.esc(logic.backend_note(backend))}</div>', padding=False)
                        ask_run = gr.Button("Run", variant="primary", size="md")
                        ask_schema = gr.HTML(logic.schema_view("retail"), padding=False)
                    with gr.Column(scale=7, min_width=320):
                        ask_out = gr.HTML(logic.initial_ask_page(backend), padding=False, elem_classes="fa-out-area")

            # --------------------------------------------------------- Playground
            with gr.Tab("SQL playground"):
                stats0, table0 = logic.playground_view("retail", logic.EXAMPLE_PAIRS[0]["cheap"])
                with gr.Column(elem_classes="fa-toolbar"):
                    with gr.Row():
                        pg_db = gr.Dropdown(logic.DOMAINS, value="retail", label="Database", scale=1, min_width=180)
                        pg_pair = gr.Dropdown(logic.pair_choices(), value="0", scale=3, min_width=240,
                                              label="Example pair (same answer, two costs; loads the cheap query)")
                    pg_sql = gr.Code(logic.EXAMPLE_PAIRS[0]["cheap"], language="sql", label="Query (one read-only SELECT)",
                                     lines=5, max_lines=14, interactive=True, show_line_numbers=False,
                                     elem_classes="fa-code-editor")
                    with gr.Row(elem_classes="fa-actions"):
                        pg_run = gr.Button("Run query", variant="primary", size="md", scale=0)
                        pg_both = gr.Button("Compare cheap and costly", variant="secondary", size="md", scale=0)
                    pg_stats = gr.HTML(stats0, padding=False)
                pg_table = gr.HTML(table0, padding=False)
                gr.HTML('<div class="fa"><h3 class="fa-section-title" style="margin-top:32px">Cheap and costly, side by side</h3></div>',
                        padding=False)
                pg_pair_out = gr.HTML(logic.run_pair("0"), padding=False)

            # ------------------------------------------------------------ Results
            with gr.Tab("Results") as res_tab:
                rv = logic.results_view("all")
                metric0 = logic.default_metric(rv)
                with gr.Column(elem_classes="fa-toolbar"):
                    with gr.Row():
                        res_split = gr.Radio(logic.split_options(logic.results_split_choices()), value="all", label="Test set",
                                             scale=3, min_width=280, elem_classes="fa-seg")
                        res_metric = gr.Dropdown(rv["metrics"], value=metric0, label="Chart metric", scale=2, min_width=200)
                    res_note = gr.HTML(render.notice_html(rv["note"], "info"), padding=False)
                gr.HTML(legend, padding=False)
                res_head = gr.HTML(rv["headline"], padding=False)
                res_plot = gr.HTML(logic.chart_html(rv, metric0), padding=False)
                gr.HTML('<div class="fa"><h3 class="fa-section-title" style="margin-top:24px">Key metrics</h3></div>', padding=False)
                res_table = gr.HTML(make_table(rv, False), padding=False)
                res_table_full = gr.HTML(make_table(rv, True), padding=False)
                res_detail = gr.HTML(rv["detail"], padding=False)

        gr.HTML(render.footer_html(EVENT, base_model_name(), footer_links()), padding=False)

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
                gr.Dropdown(choices=s["tasks"], value=s["task"], label=logic.task_info(len(s["tasks"]))),
            )

        def cb_filter(sp, a, b, dom, lv, cur):
            choices, pick = logic.on_filter_change(sp, a, b, dom, lv, cur)
            return gr.Dropdown(choices=choices, value=pick, label=logic.task_info(len(choices)))

        def cb_random(sp, a, b, dom, lv):
            pick = logic.random_task(sp, a, b, dom, lv)
            return gr.Dropdown(value=pick) if pick else gr.skip()

        def cb_diff(sp, a, b, dom, lv):
            pick = logic.most_different_task(sp, a, b, dom, lv)
            return gr.Dropdown(value=pick) if pick else gr.skip()

        quiet = {"show_progress": "minimal"}
        cmp_inputs = [split, model_a, model_b, task]
        filter_inputs = [split, model_a, model_b, domain, level, task]
        split.input(cb_split, split, [model_a, model_b, domain, level, task], api_name="compare_split", concurrency_limit=4,
                    **quiet).then(cb_view, cmp_inputs, replay_out, api_name=False, **quiet)
        for ctl in (domain, level, model_a, model_b):
            ctl.input(cb_filter, filter_inputs, task, api_name=False, concurrency_limit=4, **quiet
                      ).then(cb_view, cmp_inputs, replay_out, api_name=False, **quiet)
        task.input(cb_view, cmp_inputs, replay_out, api_name="replay", concurrency_limit=4, **quiet)
        btn_random.click(cb_random, [split, model_a, model_b, domain, level], task, api_name="random_task", concurrency_limit=4,
                         **quiet).then(cb_view, cmp_inputs, replay_out, api_name=False, **quiet)
        btn_diff.click(cb_diff, [split, model_a, model_b, domain, level], task, api_name="differ_most", concurrency_limit=4,
                       **quiet).then(cb_view, cmp_inputs, replay_out, api_name=False, **quiet)

        def cb_ask_db(db):
            ex = logic.example_choices(db)
            first = ex[0][1] if ex else None
            return logic.schema_view(db), gr.Dropdown(choices=ex, value=first), logic.question_for_example(db, first)

        ask_db.input(cb_ask_db, ask_db, [ask_schema, ask_example, ask_question], api_name=False, concurrency_limit=4)
        ask_example.input(logic.question_for_example, [ask_db, ask_example], ask_question, api_name=False, concurrency_limit=4)

        def cb_ask_run(db, q, pol):
            yield from logic.ask_run(db, q, pol, backend)

        def busy(text):
            return lambda: gr.Button(value=text, interactive=False)

        def idle(text):
            return lambda: gr.Button(value=text, interactive=True)

        ask_run.click(busy("Running..."), None, ask_run, api_name=False, queue=False).then(
            cb_ask_run, [ask_db, ask_question, ask_policy], ask_out, api_name="ask_run",
            concurrency_limit=2, time_limit=180, show_progress="hidden").then(idle("Run"), None, ask_run, api_name=False,
                                                                                queue=False)

        pg_pair.input(logic.load_pair, pg_pair, [pg_db, pg_sql], api_name=False, concurrency_limit=4)
        pg_run.click(busy("Running..."), None, pg_run, api_name=False, queue=False).then(
            logic.playground_view, [pg_db, pg_sql], [pg_stats, pg_table], api_name="playground_run", concurrency_limit=3,
            **quiet).then(idle("Run query"), None, pg_run, api_name=False, queue=False)
        pg_both.click(logic.run_pair, pg_pair, pg_pair_out, api_name="playground_pair", concurrency_limit=2, **quiet)

        def cb_results(sp, metric):
            v = logic.results_view(sp)
            m = metric if metric in v["metrics"] else logic.default_metric(v)
            return (
                render.notice_html(v["note"], "info"),
                v["headline"],
                gr.Dropdown(choices=v["metrics"], value=m),
                make_plot(v, m),
                make_table(v, False),
                make_table(v, True),
                v["detail"],
            )

        res_outputs = [res_note, res_head, res_metric, res_plot, res_table, res_table_full, res_detail]
        res_split.input(cb_results, [res_split, res_metric], res_outputs, api_name="results", concurrency_limit=4, **quiet)
        # Read the results again when the tab opens, so a new results.json shows without a restart.
        res_tab.select(cb_results, [res_split, res_metric], res_outputs, api_name=False, concurrency_limit=4,
                       show_progress="hidden")

        def cb_metric(sp, metric):
            return make_plot(logic.results_view(sp), metric)

        res_metric.input(cb_metric, [res_split, res_metric], res_plot, api_name=False, concurrency_limit=4, **quiet)

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
        server_name=args.host, server_port=args.port, theme=THEME, css=CSS, head=HEAD,
        max_threads=16, ssr_mode=False, show_error=False, share=False, footer_links=[],
    )


if __name__ == "__main__":
    main()
