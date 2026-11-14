# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # AgentTune pinned to commit 1dfd054. The tarball URL needs no git in the job image.
#     "agenttune @ https://github.com/Lexsi-Labs/AgentTune/archive/1dfd054ce59e1c1bdd566cb7e5826f836b1c8c2e.tar.gz",
#     "huggingface_hub>=1.0",
# ]
# ///
"""One Hugging Face Jobs entry point for every Frugal Analyst GPU step.

It (1) fetches this project's code, (2) runs scripts/gen_data.py, (3) runs one
step, and (4) uploads the step's outputs and log to a Hub repo, even when the
step fails. It never pushes anywhere else.

Steps:
    baseline   zero-shot runner over the test splits        -> traces
    sft        scripts/train_sft.py                          -> LoRA adapter
    grpo       scripts/train_grpo.py --init-adapter ...      -> LoRA adapter
    eval       runner with one or more adapters merged       -> traces

Code source, one of:
    --code-repo USER/NAME        a HF dataset repo holding this repo's files (preferred)
    --code-git-url URL           a git URL (needs git in the job image)

Adapter references (--init-adapter, --adapters) are a local path or
REPO_ID:SUBDIR, for example `me/frugal-runs:sft-r16/adapter`.

Step arguments go in one string: `--extra "--max-steps 30 --num-generations 4"`.
`hf jobs uv run` drops a bare `--`, so `--` works only when you run this file directly.

See scripts/jobs/README.md for the exact launch commands and costs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("FRUGAL_WORK", "/tmp/frugal"))

# --smoke: enough to prove install, gated download, the step's code path and the upload.
SMOKE_ARGS = {
    "baseline": ["--limit", "5"],
    "eval": ["--limit", "5"],
    "sft": ["--limit", "16", "--max-steps", "5"],
    "grpo": ["--limit", "8", "--max-steps", "2", "--num-generations", "2", "--batch-size", "2",
             "--grad-accum", "1", "--max-tool-steps", "3"],
}


def log(msg: str) -> None:
    print(f"[job] {msg}", flush=True)


def fetch_code(args) -> Path:
    """Download the project into WORK/code and return the path."""
    code = WORK / "code"
    if code.exists():
        shutil.rmtree(code)
    if args.code_repo:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.code_repo, repo_type="dataset", revision=args.code_revision,
            local_dir=str(code), ignore_patterns=[".venv/*", "outputs/*", "data/dbs/*", ".git/*"],
        )
    elif args.code_dir:  # local test of this script
        shutil.copytree(args.code_dir, code, ignore=shutil.ignore_patterns(".venv", "outputs", ".git", "dbs", "__pycache__"))
    elif args.code_git_url:
        cmd = ["git", "clone", "--depth", "1", args.code_git_url, str(code)]
        if args.code_revision:
            cmd[3:3] = ["--branch", args.code_revision]
        subprocess.run(cmd, check=True)
    else:
        raise SystemExit("pass --code-repo, --code-git-url or --code-dir")
    return code


def resolve_ref(ref: str, repo_type: str) -> str:
    """A local adapter path for `ref` (a path, or REPO_ID:SUBDIR on the Hub)."""
    if Path(ref).exists() or ":" not in ref:
        return ref
    from huggingface_hub import snapshot_download

    repo_id, sub = ref.split(":", 1)
    sub = sub.strip("/")
    local = WORK / "refs" / repo_id.replace("/", "__")
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(local), allow_patterns=[f"{sub}/*"])
    path = local / sub
    if not (path / "adapter_config.json").exists():
        raise SystemExit(f"no adapter_config.json under {ref}")
    return str(path)


def run(cmd: list[str], cwd: Path, logfile: Path) -> int:
    """Run a command, stream its output to stdout and to a log file."""
    log("$ " + " ".join(cmd))
    with logfile.open("a", encoding="utf-8") as f, subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc:
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def upload(run_dir: Path, args) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.output_repo, repo_type=args.output_repo_type, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir), repo_id=args.output_repo, repo_type=args.output_repo_type,
        path_in_repo=args.run_name, commit_message=f"frugal-analyst {args.step}: {args.run_name}",
    )
    log(f"uploaded {run_dir} to {args.output_repo}/{args.run_name}")


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", required=True, choices=["baseline", "sft", "grpo", "eval"])
    p.add_argument("--code-repo", default=os.environ.get("FRUGAL_CODE_REPO"))
    p.add_argument("--code-git-url", default=os.environ.get("FRUGAL_CODE_GIT_URL"))
    p.add_argument("--code-revision", default=os.environ.get("FRUGAL_CODE_REVISION"))
    p.add_argument("--code-dir", default=None, help="local copy of the code (for testing this script)")
    p.add_argument("--no-upload", action="store_true", help="keep outputs local (for testing this script)")
    p.add_argument("--output-repo", required=True, help="Hub repo for outputs, e.g. me/frugal-runs")
    p.add_argument("--output-repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--run-name", required=True, help="folder name inside the output repo")
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    # `hf jobs uv run` drops a bare `--`, so step arguments can also come as one string.
    p.add_argument("--extra", default="", help='arguments for the step script as one string, e.g. --extra "--epochs 2 --lr 2e-4"')
    p.add_argument("--label", default=None, help="trace label for baseline/eval (zero-shot, distilled, grpo, grpo-ungated)")
    p.add_argument("--splits", default="test_id,test_ood", help="task splits for baseline/eval")
    p.add_argument("--adapters", default=None, help="eval: comma list of adapter refs, merged in order")
    p.add_argument("--init-adapter", default=None, help="grpo: SFT adapter ref")
    p.add_argument("--smoke", action="store_true", help="tiny settings that cost a few cents; args after -- still win")
    args = p.parse_args(argv)
    import shlex
    extra = extra + shlex.split(args.extra)
    if args.smoke:
        # Put the smoke settings first so anything after `--` overrides them
        # (argparse keeps the last value of a repeated option).
        extra = SMOKE_ARGS[args.step] + extra
        if args.step in ("baseline", "eval"):
            args.splits = "test_id"

    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    code = fetch_code(args)
    run_dir = WORK / "run" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logfile = run_dir / "job.log"
    py = sys.executable
    status = 1
    try:
        if run([py, "scripts/gen_data.py", "--seed", "0"], code, logfile):
            raise SystemExit("gen_data failed")
        if args.step in ("baseline", "eval"):
            label = args.label or ("zero-shot" if args.step == "baseline" else "eval")
            adapters = []
            if args.step == "eval":
                if not args.adapters:
                    raise SystemExit("eval needs --adapters")
                adapters = [resolve_ref(a, args.output_repo_type) for a in args.adapters.split(",")]
            status = 0
            for split in args.splits.split(","):
                cmd = [py, "-m", "frugal_analyst.runner", "--model", args.model,
                       "--tasks", f"data/tasks/{split}.jsonl",
                       "--out", str(run_dir / "traces" / f"{label}_{split}.jsonl"), "--label", label]
                if adapters:
                    cmd += ["--adapter", ",".join(adapters)]
                status = status or run(cmd + extra, code, logfile)
        elif args.step == "sft":
            status = run([py, "scripts/train_sft.py", "--model", args.model, "--out", str(run_dir / "adapter")] + extra,
                         code, logfile)
        elif args.step == "grpo":
            cmd = [py, "scripts/train_grpo.py", "--model", args.model, "--out", str(run_dir / "adapter")]
            if args.init_adapter:
                cmd += ["--init-adapter", resolve_ref(args.init_adapter, args.output_repo_type)]
            status = run(cmd + extra, code, logfile)
    finally:
        (run_dir / "job_info.txt").write_text(
            f"step={args.step}\nstatus={status}\nseconds={time.time() - started:.0f}\nargs={sys.argv[1:]}\n"
        )
        # Checkpoints are large and the final adapter is saved anyway.
        for ckpt in run_dir.glob("adapter/checkpoint-*"):
            shutil.rmtree(ckpt, ignore_errors=True)
        if args.no_upload:
            log(f"outputs kept in {run_dir} (--no-upload)")
        else:
            upload(run_dir, args)
    log(f"done: status={status} in {time.time() - started:.0f}s")
    sys.exit(status)


if __name__ == "__main__":
    main()
