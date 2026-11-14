# Frugal Analyst demo app

A Gradio app with four tabs: Compare (replay), Ask (live), SQL playground and Results.
It needs no GPU and no model weights.

## Run it

From the repo root:

    .venv/bin/python app/app.py

Open http://127.0.0.1:7860. Use `--port` and `--host` to change them.
The app builds the SQLite files on first start if they are missing.

## What it reads

| Path | Use |
| --- | --- |
| `outputs/traces/<label>_<split>.jsonl` | Replay and trace-based numbers. Label is `zero-shot`, `distilled`, `grpo`, `grpo-ungated`, `oracle`, `naive`. Split is `test_id`, `test_ood`, `val` or `train`. |
| `data/tasks/{val,test_id,test_ood}.jsonl` | Example questions for the Ask tab. |
| `outputs/eval/results.json` | Headline numbers and charts. If it is missing or unreadable, the Results tab computes numbers from the traces and says so. |

A new trace file that follows the naming above shows up after the app restarts.
The default pair in Compare is `zero-shot` against `grpo` when both exist. Else it is the worst and best model by mean reward.

## Live model (optional)

The Ask tab shows a model policy only when a backend is available. That needs all of:

- `FRUGAL_LIVE_MODEL=1` in the environment,
- `torch`, `transformers` and `agenttune` installed,
- `frugal_analyst.runner` importable.

Optional: `FRUGAL_MODEL_ID` (default `CohereLabs/tiny-aya-global`), `FRUGAL_ADAPTER` (comma separated adapter dirs), `FRUGAL_MODEL_LABEL` (name in the UI).
Without these, the tab says that live inference is off and replay is on.

## Footer links (optional)

The footer shows a link only when its variable holds a plain `https://` URL. Unset links are hidden.

| Variable | Link text |
| --- | --- |
| `FRUGAL_LINK_GITHUB` | GitHub repository |
| `FRUGAL_LINK_MODELS` | Model collection |
| `FRUGAL_LINK_DATASETS` | Dataset collection |
| `FRUGAL_LINK_WRITEUP` | Write-up |

The footer also names the base model from `FRUGAL_MODEL_ID` (default `CohereLabs/tiny-aya-global`).

## Hugging Face Space

Use the Gradio SDK. The repo root holds the whole project, so set `app_file: app/app.py` in the Space metadata.
Install `app/requirements.txt`. The core package uses only the standard library.
The Space runtime sets the host. Paths are relative to the repo root.

## Safety

- All SQL runs through `frugal_analyst.env.execute_metered`: read-only file, one SELECT or WITH statement, row cap, output cap, VM-step abort.
- The database name comes from a fixed list and is checked on the server. Trace and task files come from fixed names found on disk. No path is built from user text.
- Question text is capped at 300 characters and SQL at 2000.
- Text from traces, models and databases is escaped before it goes into HTML. SQL colour is added per token after escaping. Long output expands with a native `<details>` element, so the page has no custom script.
- The only external request is one Google Fonts stylesheet. The page keeps its layout with system fonts if it fails.
- The queue is small, with a concurrency limit per handler. No `share=True`, no file serving, no tokens.

## Tests

    .venv/bin/python -m pytest -q tests/test_app.py

## Files

- `DESIGN.md` design principles and the reasons for them.
- `app.py` layout, theme and event wiring.
- `fa_logic.py` handler logic, no Gradio import.
- `fa_render.py` HTML and CSS.
- `fa_data.py` trace, task and results reading.
- `fa_backend.py` optional model backend.
