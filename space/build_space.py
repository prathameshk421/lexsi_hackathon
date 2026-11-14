"""Assemble a folder that uploads as a Hugging Face Gradio Space. Standard library only.

Run from the repo root:

    python space/build_space.py --out outputs/space_build \
        --github https://github.com/... --models https://huggingface.co/collections/... \
        --datasets https://huggingface.co/collections/... --writeup https://...

The folder keeps the repo layout (app/, frugal_analyst/, data/tasks/, outputs/traces/, outputs/eval/),
so the app finds its files with the same relative paths as in the repo. The Space runs `app/app.py`.
It needs no GPU, no model weights and no secret: it replays saved traces, and it builds the
synthetic SQLite databases at start-up (frugal_analyst.db.ensure_dbs, a few seconds).
The link options go into app/links.json, which the app reads when the environment variables are unset.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "outputs" / "space_build"

TITLE = "Frugal Analyst"
SHORT_DESCRIPTION = "Replay of Tiny Aya SQL agent runs and their cost"  # the Hub allows at most 60 characters
APP_FILE = "app/app.py"
PYTHON_VERSION = "3.12"  # pyproject.toml asks for 3.12 or later

APP_MODULES = ("app.py", "fa_backend.py", "fa_data.py", "fa_logic.py", "fa_render.py")
TASK_FILES = ("val.jsonl", "test_id.jsonl", "test_ood.jsonl")  # fa_data.EXAMPLE_SPLITS
TRACE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}_(test_id|test_ood|val|train)\.jsonl$")  # the name rule in fa_data
RESULT_FILES = (
    "outputs/eval/results.json",  # required
    "outputs/eval_vs_sft/results.json",  # optional: GRPO against SFT
    "outputs/eval_mixed/results.json",  # optional: the mixed-teacher experiment
)
LINK_OPTIONS = {
    "github": "FRUGAL_LINK_GITHUB",
    "models": "FRUGAL_LINK_MODELS",
    "datasets": "FRUGAL_LINK_DATASETS",
    "writeup": "FRUGAL_LINK_WRITEUP",
}
SAFE_URL = re.compile(r"^https://[A-Za-z0-9.-]+(?::\d{1,5})?(?:/[^\s\"'<>\\`]*)?$")  # same rule as fa_render.safe_url
# A file in the root of a previous build. The script deletes an old folder only when it has this file.
MARKER = ".frugal_space_build"


class BuildError(Exception):
    pass


def gradio_version(repo: Path) -> str:
    """The pinned gradio version from app/requirements.txt. It must match the installed one, if any."""
    text = (repo / "app" / "requirements.txt").read_text(encoding="utf-8")
    m = re.search(r"^gradio==([0-9][0-9A-Za-z.]*)\s*$", text, re.M)
    if not m:
        raise BuildError("app/requirements.txt has no 'gradio==<version>' line")
    pinned = m.group(1)
    try:
        installed = importlib.metadata.version("gradio")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed is not None and installed != pinned:
        raise BuildError(f"gradio {installed} is installed, but app/requirements.txt pins {pinned}")
    return pinned


def pandas_requirement() -> str:
    """The app builds the playground table with pandas. Gradio installs it too; this pins the tested version."""
    try:
        return f"pandas=={importlib.metadata.version('pandas')}"
    except importlib.metadata.PackageNotFoundError:
        return "pandas"


def check_links(links: dict[str, str]) -> dict[str, str]:
    out = {}
    for var, url in links.items():
        if not url:
            continue
        if len(url) > 300 or not SAFE_URL.match(url):
            raise BuildError(f"{var}: not a plain https URL: {url!r}")
        out[var] = url
    return out


def front_matter(sdk_version: str) -> str:
    lines = [
        "---",
        f"title: {TITLE}",
        "sdk: gradio",
        f'sdk_version: "{sdk_version}"',
        f'python_version: "{PYTHON_VERSION}"',
        f"app_file: {APP_FILE}",
        "pinned: false",
        f"short_description: {SHORT_DESCRIPTION}",
        "---",
    ]
    return "\n".join(lines) + "\n"


def space_readme(sdk_version: str, traces: list[str], links: dict[str, str]) -> str:
    link_lines = "".join(f"- {var}: {url}\n" for var, url in links.items()) or "- none set\n"
    return front_matter(sdk_version) + f"""
# Frugal Analyst demo

This Space replays saved runs of Tiny Aya Global (`CohereLabs/tiny-aya-global`) on analytics questions over SQLite.
It compares the base model, the SFT model and the SFT + GRPO model task by task, with the cost of each run.

- No model runs here. There is no GPU, no model weight and no secret.
- The Compare tab replays {len(traces)} saved trace files.
- The SQL playground runs read-only queries on synthetic databases. The app builds them at start-up.
- The Results tab reads the AuditKit results files in `outputs/`.

This folder was made by `space/build_space.py` in the project repository. Do not edit it here. Change the repo and build it again.

Footer links baked into `app/links.json`:

{link_lines}"""


def requirements() -> str:
    return (
        "# gradio comes from sdk_version in README.md. The core package uses the standard library only.\n"
        f"{pandas_requirement()}\n"
    )


def _copy(src: Path, dst: Path, files: list[str], out: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    files.append(dst.relative_to(out).as_posix())


def prepare_out(out: Path, repo: Path) -> None:
    out = out.resolve()
    if out == repo or repo.is_relative_to(out) or out == Path(out.anchor):
        raise BuildError(f"refusing to build into {out}")
    if out.exists():
        if not out.is_dir():
            raise BuildError(f"{out} exists and is not a folder")
        if any(out.iterdir()):
            if not (out / MARKER).is_file():
                raise BuildError(f"{out} is not empty and is not an old build. Pick another --out.")
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def build(out: Path, links: dict[str, str] | None = None, repo: Path = REPO_ROOT) -> list[str]:
    """Write the Space folder. Returns the list of files, relative to `out`."""
    repo = Path(repo).resolve()
    out = Path(out)
    links = check_links(links or {})
    sdk_version = gradio_version(repo)
    trace_dir = repo / "outputs" / "traces"
    traces = sorted(p.name for p in trace_dir.iterdir() if TRACE_RE.match(p.name) and p.is_file()) if trace_dir.is_dir() else []
    if not traces:
        raise BuildError(f"no trace files named <label>_<split>.jsonl in {trace_dir}")
    if not (repo / RESULT_FILES[0]).is_file():
        raise BuildError(f"missing {RESULT_FILES[0]}. Run the evaluation first.")
    prepare_out(out, repo)
    out = out.resolve()
    files: list[str] = []

    for name in APP_MODULES:
        _copy(repo / "app" / name, out / "app" / name, files, out)
    for src in sorted((repo / "frugal_analyst").glob("*.py")):
        _copy(src, out / "frugal_analyst" / src.name, files, out)
    for name in TASK_FILES:
        _copy(repo / "data" / "tasks" / name, out / "data" / "tasks" / name, files, out)
    for name in traces:
        _copy(trace_dir / name, out / "outputs" / "traces" / name, files, out)
    for rel in RESULT_FILES:
        if (repo / rel).is_file():
            _copy(repo / rel, out / rel, files, out)

    generated = {
        "app/links.json": json.dumps(links, indent=2, sort_keys=True) + "\n",
        "README.md": space_readme(sdk_version, traces, links),
        "requirements.txt": requirements(),
        MARKER: "Made by space/build_space.py. The script may delete this folder on the next build.\n",
    }
    for rel, text in generated.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        files.append(rel)
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Assemble the Hugging Face Space folder for the demo.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default: outputs/space_build)")
    for opt, var in LINK_OPTIONS.items():
        p.add_argument(f"--{opt}", default="", help=f"https URL for the footer link ({var})")
    args = p.parse_args(argv)
    links = {var: getattr(args, opt) for opt, var in LINK_OPTIONS.items()}
    try:
        files = build(args.out, links)
    except (BuildError, OSError) as exc:
        print(f"build_space: {exc}", file=sys.stderr)
        return 2
    out = args.out.resolve()
    size = sum((out / f).stat().st_size for f in files)
    print(f"Wrote {len(files)} files to {out} ({size / 1e6:.1f} MB).")
    for f in files:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
