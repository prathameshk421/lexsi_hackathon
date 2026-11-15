# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=1.0"]
# ///
"""Copy the trained LoRA adapters from the private runs repo to public model repos.

It runs as a CPU job on Hugging Face, so the 120 MB weight files never pass
through a laptop connection. It copies only the adapter weights and small config
files. It leaves out the tokenizer files (they belong to the gated base model)
and `training_args.bin` (a pickle file).

Usage (from the repo root):
    hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN scripts/jobs/publish_adapters.py \
        --source YOU/frugal-analyst-runs --owner YOU
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# run folder in the source repo -> public repo name
ADAPTERS = {
    "sft-rlsplit": "frugal-analyst-tiny-aya-sft",
    "grpo-rlsplit": "frugal-analyst-tiny-aya-grpo",
    "grpo-ungated": "frugal-analyst-tiny-aya-grpo-ungated",
    "sft": "frugal-analyst-tiny-aya-sft-all-templates",
    "sft-mixed": "frugal-analyst-tiny-aya-sft-mixed-teacher",
    "grpo-mixed": "frugal-analyst-tiny-aya-grpo-mixed-teacher",
}
KEEP = [
    "adapter_config.json", "adapter_model.safetensors", "frugal_sft_meta.json", "frugal_grpo_meta.json",
    "grpo_training_config.yaml", "lexsi_provenance.json", "training_stats.json", "log_history.jsonl",
    "reward_log.jsonl",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="private model repo with the run folders")
    p.add_argument("--owner", required=True, help="user or org that gets the public repos")
    p.add_argument("--only", action="append", default=[], metavar="RUN=NAME",
                   help="publish only these runs, for example grpo-long=frugal-analyst-tiny-aya-grpo-long")
    args = p.parse_args()
    api = HfApi()
    adapters = dict(x.split("=", 1) for x in args.only) if args.only else ADAPTERS
    for run, name in adapters.items():
        repo_id = f"{args.owner}/{name}"
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(repo_id=args.source, repo_type="model", local_dir=tmp,
                              allow_patterns=[f"{run}/adapter/{f}" for f in KEEP])
            folder = Path(tmp) / run / "adapter"
            files = sorted(f.name for f in folder.iterdir())
            if "adapter_model.safetensors" not in files:
                raise SystemExit(f"no adapter weights under {run}/adapter")
            api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
            api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                              commit_message=f"Adapter from run {run}")
            print(f"[publish] {run} -> {repo_id}: {files}", flush=True)


if __name__ == "__main__":
    main()
