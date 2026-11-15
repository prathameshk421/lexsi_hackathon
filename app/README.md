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
| `outputs/traces/<label>_<split>.jsonl` | Replay and trace-based numbers. Split is `test_id`, `test_ood`, `val` or `train`. |
| `data/tasks/{val,test_id,test_ood}.jsonl` | Example questions for the Ask tab. |
| `outputs/eval/results.json` | Headline numbers, chart, tables and template groups. If it is missing or unreadable, the Results tab computes numbers from the traces and says so. |
| `outputs/eval_vs_sft/results.json`, `outputs/eval_mixed/results.json` | Optional. Rows for "What GRPO adds on top of SFT" and the mixed-teacher run in the Key comparisons table. |

A new trace file that follows the naming above shows up after the app restarts.

## Run names

`RUN_INFO` in `fa_data.py` is the one place for run names. It holds a plain name, a short name and a one-line note per label.
It also sets the display order: baseline, SFT, GRPO, experiments, references. An unknown label is shown as it is.
Dropdowns show `plain name [label]`. The Compare and Results tabs have a "What are these runs" disclosure.

The default pair in Compare is `zero-shot` against `grpo`, else `zero-shot` against `distilled`.
Else it is the worst and best main model by mean reward.

## How the numbers are shown

- Efficiency counts only for a correct answer in the reward. A wrong run shows "n/a" for efficiency, an outline cost bar and "no correct answer" next to its ratio.
- The Results chart opens on task success. The choices start with task success, correct and optimal, tool calls and latency.
- "Efficiency, correct answers only" comes before the raw mean. The raw mean is last and its caption says that it counts wrong answers.
- The headline compares the base model with the final model on each test set. Each card shows the 95% interval, the AuditKit p-value and the verdict from `results.json`, for example "no clear change". Latency has no interval, and the card says so.

## Live model (optional)

The Ask tab shows a model policy only when a backend is available. That needs all of:

- `FRUGAL_LIVE_MODEL=1` in the environment,
- `torch`, `transformers` and `agenttune` installed,
- `frugal_analyst.runner` importable.

Optional: `FRUGAL_MODEL_ID` (default `CohereLabs/tiny-aya-global`), `FRUGAL_ADAPTER` (comma separated adapter dirs), `FRUGAL_MODEL_LABEL` (name in the UI).
Without these, the tab says at the top that live inference is off, and points to the Compare tab.
The Run button still works: it runs the oracle and naive scripts on the example questions.

## Footer links (optional)

The footer shows a link only when its variable holds a plain `https://` URL. Unset links are hidden.
If a variable is unset, the app reads it from `app/links.json`. Only `space/build_space.py` writes that file.

| Variable | Link text |
| --- | --- |
| `FRUGAL_LINK_GITHUB` | GitHub repository |
| `FRUGAL_LINK_MODELS` | Model collection |
| `FRUGAL_LINK_DATASETS` | Dataset collection |
| `FRUGAL_LINK_WRITEUP` | Write-up |

The footer also names the base model from `FRUGAL_MODEL_ID` (default `CohereLabs/tiny-aya-global`).

## Hugging Face Space

`space/build_space.py` (standard library) makes a folder that uploads as a Gradio Space with no edits. From the repo root:

    .venv/bin/python space/build_space.py --out outputs/space_build \
        --github <https URL> --models <https URL> --datasets <https URL> --writeup <https URL>

Every link option is optional. The folder keeps the repo layout, so the app finds its files with the same relative paths:
`app/` (code only), `frugal_analyst/`, `data/tasks/{val,test_id,test_ood}.jsonl`, `outputs/traces/<label>_<split>.jsonl`, the results files, `app/links.json`, `README.md` and `requirements.txt`.

- `README.md` has the Space front matter: `sdk: gradio`, `sdk_version` from `app/requirements.txt` (the script stops if the installed gradio differs), `app_file: app/app.py`, `pinned: false`. It makes no licence claim.
- `requirements.txt` pins only pandas. Gradio comes from `sdk_version`.
- The Space needs no GPU, no model weight and no secret. Live inference stays off, because torch is not installed.
- The SQLite files are not in the folder. `logic.init()` builds them at start-up in `data/dbs/` (a few seconds).
- The script deletes an old output folder only if it holds the `.frugal_space_build` marker file.

Build again right before the upload. A local test run adds `data/dbs/` and `__pycache__/` to the folder.

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
