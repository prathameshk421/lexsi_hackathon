# Submission links

Replace each placeholder with the final URL. Then do the signed-out check below. A link that is not yet public is not final.

## Links for the form

| Form field (page) | Required | Value |
|---|---|---|
| HF model collection link (page 2) | yes | https://huggingface.co/collections/prathameshk421/frugal-analyst-models-and-adapters-6af8aee6fad812f5859dc8f9 |
| HF dataset collection link (page 2) | yes | https://huggingface.co/collections/prathameshk421/frugal-analyst-datasets-traces-and-logs-6af8aee8307f37a24ac0d10d |
| Hugging Face article link (page 2) | yes | `{{HF_ARTICLE}}`. If HF articles are not open to us, the submit page accepts the GitHub README: `https://github.com/prathameshk421/lexsi_hackathon#readme` |
| GitHub repo link (page 2) | yes | https://github.com/prathameshk421/lexsi_hackathon |

## What each collection must hold

| Collection | Repos and content |
|---|---|
| Models (https://huggingface.co/collections/prathameshk421/frugal-analyst-models-and-adapters-6af8aee6fad812f5859dc8f9) | 7 adapter repos, one per run: [`grpo-long`](https://huggingface.co/aarush323/frugal-analyst-tiny-aya-grpo-long) (final, 200 GRPO steps, under the teammate account `aarush323`), [`grpo`](https://huggingface.co/prathameshk421/frugal-analyst-tiny-aya-grpo) (50 steps), [`distilled`](https://huggingface.co/prathameshk421/frugal-analyst-tiny-aya-sft), [`grpo-ungated`](https://huggingface.co/prathameshk421/frugal-analyst-tiny-aya-grpo-ungated), [`distilled-all`](https://huggingface.co/prathameshk421/frugal-analyst-tiny-aya-sft-all-templates), [`distilled-mixed`](https://huggingface.co/prathameshk421/frugal-analyst-tiny-aya-sft-mixed-teacher), [`grpo-mixed`](https://huggingface.co/prathameshk421/frugal-analyst-tiny-aya-grpo-mixed-teacher). Each model card has a loading note: merge the SFT adapter first, then the GRPO adapter. The `grpo-long` card says: merge `prathameshk421/frugal-analyst-tiny-aya-sft`, then `prathameshk421/frugal-analyst-tiny-aya-grpo`, then `aarush323/frugal-analyst-tiny-aya-grpo-long`. No base weights. |
| Datasets (https://huggingface.co/collections/prathameshk421/frugal-analyst-datasets-traces-and-logs-6af8aee8307f37a24ac0d10d) | https://huggingface.co/datasets/prathameshk421/frugal-analyst-data: task splits (`data/tasks/*.jsonl`), teacher traces (`data/teacher/*.jsonl`), all 22 evaluation trace files and their `.run.json` files (`outputs/traces/`, including the 2 of `grpo-long`), GRPO `reward_log.jsonl` and `trajectories.jsonl` of each GRPO run (the `grpo-long` files come from `aarush323/frugal-analyst-runs`), `job_info.txt` and `job.log` of each job, and `docs/results/*`. |
| Demo (optional) | `{{HF_SPACE}}` |

That is 7 adapters in total. The `grpo-long` repo lives under another account, so add it to the models collection by hand and check that the collection lists it. The smoke adapters (`smoke-sft`, `smoke2-sft`, `smoke-grpo`, `smoke2-grpo`) are optional.

## Other links in the write-up

| Link | Value |
|---|---|
| Base model | https://huggingface.co/CohereLabs/tiny-aya-global |
| AgentTune source | https://github.com/Lexsi-Labs/AgentTune (commit `1dfd054ce59e1c1bdd566cb7e5826f836b1c8c2e`) |
| AuditKit docs | https://auditkit.lexsi.ai/ |
| Job output repo used by our runs | `prathameshk421/frugal-analyst-runs` (model repo; its folders hold the adapters and logs above) |
| Job output repo used by the `grpo-long` run (teammate account) | `aarush323/frugal-analyst-runs` (folder `grpo-long`: adapter, `reward_log.jsonl`, `trajectories.jsonl`, log) |
| Job code repo used by our runs | `prathameshk421/frugal-analyst-code` (dataset repo; created private per `scripts/jobs/README.md`) |

## Check signed out

Open each link in a private browser window, not signed in to GitHub or Hugging Face.

- [ ] https://github.com/prathameshk421/lexsi_hackathon opens, shows the README, and the code is visible.
- [ ] https://huggingface.co/collections/prathameshk421/frugal-analyst-models-and-adapters-6af8aee6fad812f5859dc8f9 opens and lists the 7 adapter repos.
- [ ] Each adapter repo opens with no access request form (un-gated) and shows its model card. This includes https://huggingface.co/aarush323/frugal-analyst-tiny-aya-grpo-long.
- [ ] `aarush323/frugal-analyst-runs` is public, or its `grpo-long` files are copied into the traces dataset.
- [ ] https://huggingface.co/collections/prathameshk421/frugal-analyst-datasets-traces-and-logs-6af8aee8307f37a24ac0d10d opens and lists the traces dataset.
- [ ] https://huggingface.co/datasets/prathameshk421/frugal-analyst-data opens with no access request form, and the files can be downloaded.
- [ ] `{{HF_ARTICLE}}` (or the README) opens.
- [ ] `{{HF_SPACE}}` starts, if we link it.
- [ ] No file in any public repo holds a token or a private path with secrets.
- [ ] Each link in the form matches the link in this file, letter for letter.
