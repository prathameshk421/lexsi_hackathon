# Submission links

Replace each placeholder with the final URL. Then do the signed-out check below. A link that is not yet public is not final.

## Links for the form

| Form field (page) | Required | Value |
|---|---|---|
| HF model collection link (page 2) | yes | `{{HF_MODELS_COLLECTION}}` |
| HF dataset collection link (page 2) | yes | `{{HF_DATASETS_COLLECTION}}` |
| Hugging Face article link (page 2) | yes | `{{HF_ARTICLE}}`. If HF articles are not open to us, the submit page accepts the GitHub README: `{{GITHUB_REPO}}#readme` |
| GitHub repo link (page 2) | yes | `{{GITHUB_REPO}}` |

## What each collection must hold

| Collection | Repos and content |
|---|---|
| Models (`{{HF_MODELS_COLLECTION}}`) | `{{HF_ADAPTERS_REPO}}`: the 3 SFT adapters (`sft` for `distilled-all`, `sft-rlsplit` for `distilled`, `sft-mixed` for `distilled-mixed`) and the 3 GRPO adapters (`grpo-rlsplit`, `grpo-ungated`, `grpo-mixed`), each with a loading note: merge the SFT adapter first, then the GRPO adapter. No base weights. |
| Datasets (`{{HF_DATASETS_COLLECTION}}`) | `{{HF_TRACES_DATASET}}`: task splits (`data/tasks/*.jsonl`), teacher traces (`data/teacher/*.jsonl`), all 20 evaluation trace files and their `.run.json` files (`outputs/traces/`), GRPO `reward_log.jsonl` and `trajectories.jsonl` of each GRPO run, `job_info.txt` and `job.log` of each job, and `docs/results/*`. |
| Demo (optional) | `{{HF_SPACE}}` |

That is 6 adapters in total. The smoke adapters (`smoke-sft`, `smoke2-sft`, `smoke-grpo`, `smoke2-grpo`) are optional.

## Other links in the write-up

| Link | Value |
|---|---|
| Base model | https://huggingface.co/CohereLabs/tiny-aya-global |
| AgentTune source | https://github.com/Lexsi-Labs/AgentTune (commit `1dfd054ce59e1c1bdd566cb7e5826f836b1c8c2e`) |
| AuditKit docs | https://auditkit.lexsi.ai/ |
| Job output repo used by our runs | `prathameshk421/frugal-analyst-runs` (model repo; its folders hold the adapters and logs above) |
| Job code repo used by our runs | `prathameshk421/frugal-analyst-code` (dataset repo; created private per `scripts/jobs/README.md`) |

## Check signed out

Open each link in a private browser window, not signed in to GitHub or Hugging Face.

- [ ] `{{GITHUB_REPO}}` opens, shows the README, and the code is visible.
- [ ] `{{HF_MODELS_COLLECTION}}` opens and lists the adapters repo.
- [ ] `{{HF_ADAPTERS_REPO}}` opens with no access request form (un-gated) and lists every adapter folder.
- [ ] `{{HF_DATASETS_COLLECTION}}` opens and lists the traces dataset.
- [ ] `{{HF_TRACES_DATASET}}` opens with no access request form, and the files can be downloaded.
- [ ] `{{HF_ARTICLE}}` (or the README) opens.
- [ ] `{{HF_SPACE}}` starts, if we link it.
- [ ] No file in any public repo holds a token or a private path with secrets.
- [ ] Each link in the form matches the link in this file, letter for letter.
