# Submission checklist

The form fields are in the order of the Tally form. "Team member fills this" means only a registered member can give the value. Nobody submits the form without a final review by the team.

## Page 1

| Field | Required | Value |
|---|---|---|
| Registered email | yes | Team member fills this. Use the exact email used to register. Check it for typos. |
| Abstract (under 500 words) | yes | Paste `submission/abstract.txt`. It has 469 words (counted with a script). |
| Discord username | no | Team member fills this. |

## Page 2

| Field | Required | Value |
|---|---|---|
| HF model collection link | yes | https://huggingface.co/collections/prathameshk421/frugal-analyst-models-and-adapters-6af8aee6fad812f5859dc8f9 |
| HF dataset collection link | yes | https://huggingface.co/collections/prathameshk421/frugal-analyst-datasets-traces-and-logs-6af8aee8307f37a24ac0d10d |
| Hugging Face article link | yes | https://github.com/prathameshk421/lexsi_hackathon#readme |
| GitHub repo link | yes | https://github.com/prathameshk421/lexsi_hackathon |

## Page 3

| Field | Tick |
|---|---|
| Repos used | [x] AgentTune, [x] AuditKIT. Leave CuratorKIT, AlignTune, SafeTune, CircuitKIT and Other empty. |
| Models used | [x] Tiny Aya Global. Leave the others empty. |
| Track | [x] Track 3: Agentic Fine-tuning |
| Track 3 problem statements (at most 2) | [x] Data and database agents (6), [x] Efficient trajectories (2) |

Do not tick Agentic distillation (3) or Reward design and reward-hacking defences (9). The form allows only 2. The write-up says the work also touches them.

## Page 4

| Field | Required | Value |
|---|---|---|
| Consent: sharing by partners | no | Team member decides. |
| Consent: being tagged | no | Team member decides. |
| "All repos and collections are public and final as of Nov 15, 11:59 PM AoE. Nothing changes after the deadline." | yes | Tick only after every check below passes. |
| CAPTCHA | yes | Team member fills this. |

## Public and final checks (from the reminder mail)

- [ ] The HF models collection holds every adapter: 3 SFT and 3 GRPO (see `submission/links.md`).
- [ ] The HF datasets collection holds the task splits, teacher traces, all evaluation traces, GRPO logs and rollouts, and job logs.
- [ ] Both collections are public, and every repo inside them is public and un-gated.
- [ ] The GitHub repo is public. (The team brief says it was private when last checked.)
- [ ] The write-up is public.
- [ ] Every placeholder `{{...}}` in `README.md`, `submission/links.md` and this file is replaced.
- [ ] The registered email has no typo.
- [ ] Each link opens in a signed-out browser window (`submission/links.md`).
- [ ] No secret (for example `HF_TOKEN`) is in any public file.
- [ ] Nothing changes after Nov 15, 11:59 PM AoE (Monday Nov 16, 5:29 PM IST). Only the last submission counts.

## Write-up sections against the official layout

| Section | Required | In `README.md` |
|---|---|---|
| Title and one-line claim | yes | Title and "Claim" |
| Tracks and problem statements | yes | "Tracks and problem statements" |
| Abstract | yes | "Abstract" and `submission/abstract.txt` |
| Headline before and after | yes | "For judges: one minute", "Same model, before and after" |
| Base model and method in one line | yes | "For judges: one minute" |
| Problem statement and approach, why this problem | yes | "Problem and approach" |
| Kits used, how they connect | yes | "Hackathon toolkits used" with a diagram |
| Models and licences | yes | "Models and licences" |
| Datasets, splits, leak check, environment | yes | "Data, tasks and environment" |
| Metrics and why | yes | "Metrics" |
| Method, hyperparameters per kit | yes | "Method and experiments" |
| Results, same model before and after | yes | "Results" |
| Limitations | yes | "Limitations" |
| With more credits | yes | "With more credits" |
| Artifacts | yes | "Artifacts" |
| Documentation and reproduction | yes | "Reproduction" and `submission/reproduce.md` |
| Declaration of outside work | yes | "Declaration of work outside the hackathon" |
| Teammates tagged | yes | "Teammates" (handles: team member fills this) |
| General capability benchmark (Track 3) | optional | Not run. Stated in Limitations and With more credits. |
| Closed-model baseline, cost per 1k vs API | optional | Not run. Our own cost per 1,000 tasks is given. |
