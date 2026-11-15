# Training notes: what we verified in AgentTune

AgentTune commit `1dfd054ce59e1c1bdd566cb7e5826f836b1c8c2e`. Paths below are
relative to the AgentTune repo (`src/agenttune/...`). TRL paths are relative to
the installed `trl==1.7.1` package (`.venv/lib/python3.12/site-packages/trl/`).
Every claim here was read in the source or seen in a run on this Mac. Where a
claim is only an estimate, it says so.

## 1. Install

```bash
uv pip install --python .venv/bin/python \
  "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git@1dfd054ce59e1c1bdd566cb7e5826f836b1c8c2e"
```

Resolved: torch 2.11.0, transformers 5.23.0, trl 1.7.1, peft 0.21.3,
datasets 5.2.0, accelerate 1.15.0. It changed three packages already in the venv:

| Package | Before | After | Why |
|---|---|---|---|
| huggingface-hub | 2.4.0 | 1.33.0 | transformers 5.x needs hub < 2 |
| pillow | 13.0.0 | 11.3.0 | AgentTune pins `pillow>=11.3.0,<12` |
| python-dateutil | 4.0.0 | 2.9.0.post0 | a dependency upper bound |

No conflict needed a manual fix. `uv pip check` reports all 257 packages
compatible, and `import auditkit, gradio` works (gradio 6.32.0, auditkit 1.1.11).
The `hf` CLI in the venv is now version 1.33.0; `hf jobs uv run` exists in it.

## 2. The Tiny Aya tool format

### 2a. The chat template ignores tools, so AgentTune uses a prompt fallback

Tiny Aya's `tokenizer_config.json` template renders only `system`, `user` and
`assistant`/`chatbot` content. It never reads `tools` or `tool_calls`, and a
`tool` turn comes out as an empty `<|START_OF_TURN_TOKEN|><|END_OF_TURN_TOKEN|>`.

- `rollout_factory._template_renders_tools` (rollout_factory.py:524-539) renders
  a probe with and without `tools`. For Tiny Aya the two are equal, so it
  returns False. Verified with the real tokenizer.
- `_execute_trajectory` then appends `DEFAULT_TOOLS_FALLBACK_PROMPT`
  (rollout_factory.py:544-551) to the system turn, with `{tools}` = the JSON
  schemas, `json.dumps(..., indent=2)` (rollout_factory.py:2188-2221). It sets
  `fold_tool_results = True`.
- Tool schemas come from `transformers.utils.get_json_schema` on our plain
  functions (`_build_tool_schema`, rollout_factory.py:713-722). The Google-style
  docstrings in `frugal_analyst/env.py` become the descriptions.
- The Tiny Aya template puts the system turn under `# Developer Preamble`, after
  Cohere's fixed safety and default preambles.

### 2b. What the model must emit to call a tool

The fallback prompt asks for:

```
[{"tool_name": "<name>", "parameters": {"<argument>": <value>}}]
```

Only the list, no other text. Several calls in one list are allowed.
`_extract_tool_calls` (rollout_factory.py:1090-1289) checks the Cohere parsers
first (1108-1114). `tool_call_parse._parse_cohere_json` (tool_call_parse.py:366-384)
accepts a JSON list whose items all carry `tool_name`, and `_as_calls`
(tool_call_parse.py:31-80) normalises them to
`{"type": "function", "function": {"name", "arguments": <json string>}}`.
Other shapes (`<tool_call>{"name":..,"arguments":..}</tool_call>`, ReAct, and so
on) are also parsed, further down the same function.

A turn with no parsable call ends the episode. Its text is `final_response`.

### 2c. How tool results come back

Tools are called as `fn(**json.loads(arguments))` (`_run_tools_parallel`,
rollout_factory.py:774-789: the first `fn(**kwargs)` with a JSON string raises
TypeError, the retry loads the JSON). An exception becomes the string
`{'error': '...'}`. An unknown tool name becomes `{'error': "Tool 'x' not found"}`.

The result is added as a `tool` message, and because the template drops tool
turns, `fold_tool_messages_into_user` (rollout_factory.py:107-142) rewrites it
as a user turn in `DEFAULT_TOOL_RESULT_FORMAT` (rollout_factory.py:102-104):

```
Tool result for run_sql: COUNT(*)
24
Use this result to answer the original question.
```

Rendered with the real Tiny Aya tokenizer, the second turn's prompt ends:

```
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>[{"tool_name": "get_schema", "parameters": {"database": "retail"}}]<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Tool result for get_schema: orders(id INTEGER, ...)
Use this result to answer the original question.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>
```

The template is prefix-preserving (turn 1's prompt is a prefix of turn 2's).

### 2d. The completion the reward function receives

TRL calls the reward as `reward_func(prompts=..., completions=..., completion_ids=..., **reward_kwargs)`
(trl grpo_trainer.py:1389-1391). The completion comes from the rollout's
`completion_ids`, which `_execute_trajectory` builds by concatenating, per episode:

- each model turn's sampled token ids, up to and including the first EOS
  (rollout_factory.py:2066-2094), with mask 1;
- each tool result's raw text, `tokenizer.encode(str(result), add_special_tokens=False)`
  (rollout_factory.py:2376-2380), with mask 0. This is the raw tool output, not
  the folded "Tool result for ..." user turn.

TRL decodes them with `skip_special_tokens=True`. For a chat-format `prompt`
column (ours) TRL wraps the text as `[{"role": "assistant", "content": text}]`
(trl grpo_trainer.py:1878-1891); for a plain-string prompt it is the bare string.
So the AgentTune docs are right that the content is one flat string, and wrong
only in that a chat prompt gives a one-message list around it. Seen in our
smoke run: `type(completions[0]) = list`.

Shape of the flat text for the efficient path:

```
[{"tool_name": "get_schema", "parameters": {"database": "clinic"}}]patients(...) -- 900 rows
...[{"tool_name": "run_sql", "parameters": {"database": "clinic", "query": "SELECT COUNT(*) ..."}}]COUNT(*)
24ANSWER: 24
```

There is no separator between a tool output and the next model turn.

Every extra key of the rollout output also reaches the reward as a kwarg
(trl grpo_trainer.py:2312-2318). The keys we saw in the smoke run:
`completion_ids, conversations, log_extra, log_metric, prompts, queries,
responses, retrieved_chunk_ids, rewards, task_id, tool_call_counts, tools_used,
trainer_state, trajectories`. `task_id` is our dataset column.

**What our reward uses.** `trajectories[i]` is AgentTune's `Trajectory` for the
same rollout. Each tool step's `action["tool_calls"]` holds exactly the calls the
loop executed. `make_grpo_reward` reads these (`calls_from_trajectory`) and only
falls back to parsing the flat text (`parse_completion`) when no trajectory is
present. The text parse cannot tell an executed call from one the model wrote in
its last turn after the step limit; the trajectory can.

### 2e. Turns, tokens and the kwargs that control them

| Limit | Where it is set | Default | Our setting |
|---|---|---|---|
| Tool rounds per episode | `create_rollout_fn(max_steps=...)`; `TrlAgenticGrpo` passes `max_steps_per_turn` (agentic_grpo.py:459) | 20 | `--max-tool-steps` 4 for SFT replay, GRPO and runner (section 7c) |
| top_p, GRPO | `GRPOConfig.top_p` -> `trainer.generation_config` (trl grpo_trainer.py:912-927), passed to `generate()` (rollout_factory.py:2038-2056); AgentTune sets 0.95 (agentic_grpo.py:423) | 0.95 | `--top-p 0.95` |
| New tokens per model turn, GRPO | `getattr(trainer, "_max_new_tokens", 256)` (rollout_factory.py:2049) | 256 | `trainer._max_new_tokens = --max-completion-length` |
| Temperature, GRPO | `getattr(trainer, "_temperature", 0.7)` (rollout_factory.py:2050), `do_sample=True` always | 0.7 | `trainer._temperature = --temperature` |
| New tokens per turn, runner | `gen_cfg["max_new_tokens"]` (transformers_engine.py:466) | 512 | `--max-new-tokens 256` |
| Total completion length | nothing truncates the concatenated ids | none | bounded by turns x per-turn cap + tool output |

`GRPOConfig.max_completion_length` and `GRPOConfig.temperature` do
**not** reach the agentic rollout. TRL puts them in `trainer.generation_config`
(trl grpo_trainer.py:912-929), but AgentTune passes explicit `max_new_tokens`
and `temperature` to `generate()` next to that config, and explicit arguments
win. Nothing in `TrlAgenticGrpo` sets `_max_new_tokens` or `_temperature` (only
the DPO wrapper does, agentic/dpo/agentic_dpo.py:586-587). Our rollout function
sets both on the trainer before each call.

Number of model generations per episode is at most `max_steps + 1`: the loop
breaks after `max_steps` tool rounds, then always generates once more after the
last tool result (rollout_factory.py:2491-2521).

### 2f. LoRA target modules for Cohere2

`ensure_lora_targets` (rollout_factory.py:213-245) sets `target_modules="all-linear"`
when the LoRA config names no targets and the model type is missing from peft's
default map. In peft 0.21.3, `"cohere2"` (and `"cohere"`) is missing from
`TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`, so without explicit targets
peft would raise and AgentTune would fall back to all-linear. We name the
targets ourselves: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj` (the `nn.Linear` names in transformers' `modeling_cohere2.py`).
SmolLM2 (Llama) has the same names. `all-linear` in peft excludes `lm_head`,
which matters here: Tiny Aya ties a 262,144 x 2,048 embedding to the output.

## 3. Problems found, and what we did

1. **Double BOS (Tiny Aya).** The template writes `<BOS_TOKEN>`, and calling the
   tokenizer on the rendered text adds another. AgentTune's trainer rollout does
   that (`tokenizer(prompt_text, ...)`, rollout_factory.py:2012-2017), and so does
   TRL's SFT tokenizer for prompt/completion strings. Seen: `tok(text)` starts
   `[2, 2, 5]`, `apply_chat_template(tokenize=True)` starts `[2, 5, 9]`. But the
   GRPO training pass uses `prompt_ids` from the template (single BOS,
   rollout_factory.py:1886-1906), so the rollout sampled in a different context
   than the one it is trained in. Fix: `runner.load_tokenizer` sets
   `tok.add_bos_token = False` when the template writes the BOS itself. After
   that, `tok(text)` starts `[2, 5, 9]` too. SFT, GRPO and the runner all use it.
2. **Per-turn token cap and temperature ignored** (section 2e). Fix: set
   `trainer._max_new_tokens` and `trainer._temperature` in our rollout function.
3. **Rollouts run in train mode.** TRL calls the rollout during `training_step`,
   so the model is in train mode: LoRA dropout is on while sampling, and with
   gradient checkpointing transformers turns the KV cache off
   (`modeling_layers.py`, "Caching is incompatible with gradient checkpointing").
   Fix: our rollout function switches to `model.eval()` for generation and back.
4. **Turn context in the GRPO loss.** The training sequence is
   `prompt_ids + [turn 1 ids][raw tool text ids][turn 2 ids]...`, but turn 2 was
   sampled after a chat-formatted user turn ("Tool result for ...", with the
   template tokens). So the log-probs of turn 2 and later are computed in a
   context the model never saw. Fix (`with_chat_turn_context` in
   scripts/train_grpo.py): keep each model turn's sampled ids, and replace each
   tool span with the exact template text between that turn and the next
   (`render(conv[:next_turn], add_generation_prompt=True)` minus
   `render(conv[:this_turn + 1])`), tokenised without special-token additions,
   with mask 0. The smoke run checks that every rebuilt sequence decodes to the
   rendered chat. `--no-turn-context-fix` turns it off.
5. **AgentTune's generic distill rows use another format.** `EventLog.as_dataset_rows`
   (agentic/events.py:111-158) writes calls as
   `<tool_call>{"name": ..., "arguments": ...}</tool_call>` (events.py:139),
   tool results as `tool` turns, and no tool list in the system turn. That is
   not what Tiny Aya sees in the rollout. Fix: scripts/train_sft.py replays the
   teacher through the real rollout loop and trains on the recorded prompts
   (section 4).
6. **`TrlAgenticGrpo` drops unknown kwargs silently** (`_split_kwargs`,
   agentic_grpo.py:78-103) and its defaults differ from TRL's
   (`gradient_accumulation_steps=16`, `learning_rate=1e-6`, `warmup_steps=10`,
   `beta=0.04`, agentic_grpo.py:401-425). train_grpo.py runs `setup_data()` and
   `setup_trainer()` itself, prints the resolved `GRPOConfig`, and stops if any
   setting we passed did not take effect.
7. **Tiny Aya stop token (runner).** `generation_config.json` has
   `eos_token_id: 3` (`<EOS_TOKEN>`), but a chat turn ends with
   `<|END_OF_TURN_TOKEN|>` (id 6, the tokenizer's eos). The transformers engine
   stops on `model.generation_config.eos_token_id` (transformers_engine.py:516).
   The runner sets it to `[6, 3]`. The trainer path already stops on the
   tokenizer eos.

Not changed:

- In the trainer path AgentTune returns 0.0 for every sampled log-prob
  (rollout_factory.py:2084-2085). That is fine without vLLM: TRL only uses them
  for vLLM importance sampling.
- `_run_trajectories` runs episodes one by one (rollout_factory.py:1748-1771).
  Its threaded mode is only used without a trainer and shares one model across
  threads, so the runner does not use it. Generation is batch size 1 everywhere.
- The process-wide monkey-patch of `GRPOTrainer._generate_single_turn`
  (agentic_grpo.py:487-537) is not reached in trl 1.7.1: TRL's own `_generate`
  calls `rollout_func(prompts, self)` directly (trl grpo_trainer.py:1852-1871).
- `create_agentic_trainer("grpo", tools=..., rollout_func=...)`: when a
  `rollout_func` is passed, AgentTune does not build its own and pops `tools`
  before `GRPOTrainer` (agentic_grpo.py:451-473, 570-572), so TRL's native tool
  loop never runs.

## 4. How SFT, GRPO and the runner share one format

| | Builds the conversation | Renders the prompt |
|---|---|---|
| SFT | AgentTune `create_rollout_fn` (via `Project.collect_rollout`) with a scripted teacher engine | `_render_chat_template` |
| GRPO | AgentTune `create_rollout_fn`, trainer path | `_render_chat_template` |
| Runner | AgentTune `create_rollout_fn`, `TransformersRolloutEngine` | `tokenizer.apply_chat_template(..., tools=...)` inside the engine |

All three start from `trajectory.build_prompt_messages(task)`:
`[system: SYSTEM_PROMPT, user: "Database: ...\nQuestion: ..."]`. train_sft.py
checks that the trainer render and the engine render give the same string, and
prints `trainer render == inference engine render: True`.

SFT details. `ScriptedTeacherEngine` (scripts/train_sft.py) is an AgentTune
`RolloutEngine` whose `generate()` returns the teacher's next turn: one
`[{"tool_name": ..., "parameters": ...}]` list per oracle call, then
`ANSWER: <gold>`. `Project.collect_rollout` drives it with the real tools, so
tool outputs in the prompts are produced by the loop itself. A replay check
scores every replayed episode with `score_trajectory` and stops unless it
matches the teacher trace. Each recorded `(conversation, teacher turn)` becomes
one row: prompt = rendered conversation, completion = teacher text plus the
template's end-of-turn text, cut after the eos token
(`<|END_RESPONSE|><|END_OF_TURN_TOKEN|>` for Tiny Aya). `Project.distill` then
calls our `ExactFormatSFTFactory` (TRL `SFTTrainer`, LoRA, prompt/completion
data so the loss is on completion tokens only). 400 teacher traces give 1,346
rows (254 episodes with 3 turns, 146 with 4).

GRPO start point. `--init-adapter` merges the SFT adapter into the base
(`PeftModel.from_pretrained(...).merge_and_unload()`), then GRPO trains a fresh
LoRA. With a PEFT model TRL gets the KL reference by disabling the adapter, so
the reference is the distilled policy. If we continued the SFT adapter instead,
the reference would be the zero-shot base and `beta` would pull the policy back
toward it. The cost: inference needs both adapters, merged in order
(`--adapter sft,grpo`). train_grpo.py prints that command and writes
`inference_adapters` to `frugal_grpo_meta.json`.

## 5. Local smoke test (CPU, SmolLM2-135M-Instruct)

SmolLM2's template also ignores `tools`, so it takes the same fallback path as
Tiny Aya (AgentTune prints "The chat template of HuggingFaceTB/SmolLM2-135M-Instruct
ignores `tools`; adding the tool schemas to the system prompt ..."). This test
checks code paths, not model quality.

**SFT, 8 traces, 3 steps:**

```bash
.venv/bin/python scripts/train_sft.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --limit 8 --max-steps 3 --batch-size 2 --grad-accum 2 --out outputs/smoke/sft --device cpu
```
```
[sft] replay check: 8/8 episodes match the teacher score
[sft] trainer render == inference engine render: True
[sft] 24 rows, longest 998 tokens, 0 dropped for --max-length 3072
[sft] Project.distill handed over 8 generic rows; training on 24 exact-format rows
[sft] saved adapter to outputs/smoke/sft: {"n_rows": 24, "global_step": 3, "train_loss_first": 1.18..., "train_loss_last": 0.537...}
```

A row from `sft_rows_preview.jsonl` (turn 2; prompt tail, then completion):

```
...Use this result to answer the original question.<|im_end|>\n<|im_start|>assistant\n
COMPLETION: [{"tool_name": "run_sql", "parameters": {"database": "clinic", "query": "SELECT COUNT(*) FROM appointments WHERE appt_date >= '2023-06-01' AND appt_date < '2023-07-01' AND fee > 150"}}]<|im_end|>
```

To get a smoke model that really calls tools, a longer run (96 traces, 60
steps, lr 1e-3, 11 minutes on CPU) to `outputs/smoke/sft60`:
`replay check: 96/96`, `324 rows`, loss 0.797 to 0.0018.

**Runner, 3 tasks, with that adapter:**

```bash
.venv/bin/python -m frugal_analyst.runner --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter outputs/smoke/sft60 --tasks data/tasks/test_id.jsonl --limit 3 \
  --out outputs/smoke/traces/distilled_test_id.jsonl --label distilled --device cpu
```
```
[runner] 1/3 clinic-L1-0002 correct=False reward=0.100 calls=2 latency=3.317s
[runner] 2/3 clinic-L1-0006 correct=False reward=0.100 calls=2 latency=3.603s
[runner] 3/3 clinic-L1-0010 correct=True reward=1.000 calls=2 latency=3.84s
```

**GRPO, 2 optimizer steps on top of `sft60`:**

```bash
.venv/bin/python scripts/train_grpo.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --init-adapter outputs/smoke/sft60 --limit 8 --max-steps 2 --num-generations 4 \
  --batch-size 4 --grad-accum 1 --max-completion-length 96 --max-tool-steps 3 \
  --out outputs/smoke/grpo60 --device cpu
```

The resolved config printed all our values back (num_generations 4, lr 1e-05,
beta 0.04, temperature 0.8, max_completion_length 96, gradient_accumulation_steps 1,
use_vllm False, loss_type dapo). What the reward function received in the first batch:

```
[grpo] turn-context fix: 4/4 sequences match the rendered chat
[grpo] type(completions[0]) = list; kwargs = ['completion_ids', 'conversations', 'log_extra', 'log_metric', 'prompts', 'queries', 'responses', 'retrieved_chunk_ids', 'rewards', 'task_id', 'tool_call_counts', 'tools_used', 'trainer_state', 'trajectories']
[grpo] task_id = clinic-L1-0008
[{"tool_name": "get_schema", "parameters": {"database": "clinic"}}]user
Tool result for get_schema: patients(patient_id INTEGER, ...) -- 900 rows
...
Use this result to answer the original question.
assistant
[{"tool_name": "run_sql", "parameters": {"database": "clinic", "query": "SELECT COUNT(*) FROM appointments WHERE appt_date >= '2024-05-01' AND appt_date < '2024-06-01'"}}]user
Tool result for run_sql: COUNT(*)
141
Use this result to answer the original question.
assistant
ANSWER: 141
[grpo] parse_completion(text) -> ([ToolCall(name='get_schema', arguments={'database': 'clinic'}), ToolCall(name='run_sql', arguments={...})], 'ANSWER: 141')
[grpo] calls_from_trajectory -> (same two calls, 'ANSWER: 141')
[grpo] rewards for this batch = [0.1, 0.1, 1.0, 1.0]
```

With the turn-context fix on, the text the reward sees includes the chat glue
("user\nTool result for ...") because TRL decodes the rebuilt ids. Without the
fix it is the bare format of section 2d. Both parse the same.

`reward_log.jsonl` (one line per optimizer step):

```
{'step': 1, 'reward': 0.55, 'reward_std': 0.5196, 'frugal/correct': 0.5, 'frugal/efficiency': 0.902, 'frugal/tool_calls': 2.25, 'loss': 0.0462}
{'step': 2, 'reward': 0.0875, 'reward_std': 0.025, 'frugal/correct': 0.0, 'frugal/efficiency': 0.828, 'frugal/tool_calls': 2.25, 'loss': 0.1053}
```

About 28 s per step on CPU for 4 rollouts. The chained eval
(`--adapter outputs/smoke/sft60,outputs/smoke/grpo60`, 3 test_ood tasks) ran
and wrote 3 traces.

**Jobs entry point, locally** (`FRUGAL_DEVICE=cpu`, `--code-dir . --no-upload`):
`--step baseline` ran gen_data and the runner and wrote traces; `--step sft --smoke`
and `--step grpo --smoke --init-adapter <sft smoke adapter>` ran as well.

## 6. Real Tiny Aya, zero-shot, 3 episodes on MPS (bf16)

```bash
HF_HUB_CACHE=outputs/hf_cache .venv/bin/python -m frugal_analyst.runner \
  --model CohereLabs/tiny-aya-global --tasks data/tasks/test_id.jsonl --limit 3 \
  --out outputs/smoke/traces/aya-zero-shot_test_id.jsonl --label zero-shot --device mps
```
```
[runner] 1/3 clinic-L1-0002 correct=False reward=0.100 calls=5 latency=101.226s
[runner] 2/3 clinic-L1-0006 correct=False reward=0.100 calls=5 latency=201.834s
[runner] 3/3 clinic-L1-0010 correct=False reward=0.100 calls=2 latency=145.003s
```

What this shows:

- Tiny Aya follows the fallback format on its first turn:
  `[{"tool_name": "get_schema", "parameters": {"database": "clinic"}}]`, and
  AgentTune parses and runs it. It also wrote valid `run_sql` calls with real
  column names after it saw the schema.
- It does not stop after the call. In the same turn it writes a fake result
  and answer: `...}]\nTool result: 1\nANSWER: 1`. The call still runs, the loop
  continues, and the episode only ends at the step limit or when a turn has no
  call. So zero-shot answers are mostly guesses. The SFT data teaches the
  missing behaviour: every teacher call turn ends with the end-of-turn token
  right after the list.
- 116 to 398 generated tokens per episode. MPS speed (about 2 to 3 tokens/s
  with the engine's extra log-prob pass) says nothing about GPU speed.
- Traces were written through `build_trace`, with `latency_s` and
  `completion_tokens` filled.

## 7. After the first GRPO run: RL-only templates, guard, stop rule

The first GRPO run started from the SFT adapter trained on all 400 train
tasks. Every rollout got reward 1.0 and the gradient was 0, so we cancelled it.
test_id and train share all 38 train-domain templates, and SFT copied them.

### 7a. RL-only templates

Rule (`RL_ONLY_TEMPLATES`, `is_rl_only` in frugal_analyst/tasks.py): in each
train domain, the last template of each level in `TEMPLATES` order is RL-only.
The rule reads only the template list. It does not touch generation, so the
four split files and `data/teacher/train.jsonl` keep their task content
(tests/test_rl_split.py rebuilds them and compares bytes).

| | Count |
|---|---|
| RL-only templates | 12 (4 domains x 3 levels) |
| Train tasks from them | 126 of 400 (42 per level) |
| test_id tasks from them | 38 of 120 |
| `data/tasks/train_sft.jsonl` and `data/teacher/train_sft.jsonl` | 274 tasks (L1 84, L2 86, L3 104) |
| `data/tasks/train_rl.jsonl` | 168 = 126 RL-only + 42 SFT-seen |

The 42 SFT-seen tasks (one per 3 RL-only tasks) show if GRPO makes the model
forget. They are the train_sft tasks with the smallest `zlib.crc32(task_id)`.
7 of the 12 RL-only templates have a text (top-1) answer.

### 7b. GRPO settings and the no-signal guard

`TrlAgenticGrpo` keeps a kwarg only if it is in `GRPOConfig.__init__` or
`GRPOTrainer.__init__` (`_split_kwargs`, agentic_grpo.py:78-103). `top_p`,
`scale_rewards`, `loss_type` and `epsilon_high` are all `GRPOConfig` fields in
trl 1.7.1 (grpo_config.py:495, 748, 760, 664). train_grpo.py passes each one,
checks it in the resolved config, and also checks
`trainer.generation_config.top_p`, because the rollout samples with that config.

`frac_reward_zero_std` (trl grpo_trainer.py:2338-2359, 2401) is the share of
rows whose reward std is 0. With `scale_rewards="group"` or `"none"` it is per
group. With `"batch"` it uses the batch std, so it no longer says if a group
has signal. So the reward wrapper `with_group_spread_metric` logs our own
`frugal/frac_groups_no_spread` (groups are runs of `num_generations` rewards).
`NoSignalGuard` reads it (or TRL's metric as a fallback) from each log line.
If all of the first N steps (`--stop-if-no-signal`, default 3) are 1.0, it
sets `control.should_training_stop`. The script then writes
`frugal_grpo_meta.json` with `stopped_no_signal: true`, saves no adapter, and
exits with code 4. `reward_log.jsonl` has `reward`, `frac_reward_zero_std`,
`frugal/frac_groups_no_spread`, `frugal/correct` (share of correct
rollouts), `frugal/tool_calls` and `frugal/efficiency` per step.

Seen on CPU (SmolLM2-135M, sft60 adapter, 1 prompt x 2 generations,
`--temperature 0.01`, `--stop-if-no-signal 1`): both rollouts scored 0.1,
the guard stopped after step 1, exit code 4, with `reward_log.jsonl`,
`trajectories.jsonl` and `frugal_grpo_meta.json` on disk. The same run printed
`top_p = 0.9, scale_rewards = batch, loss_type = dr_grpo, epsilon_high = 0.28,
generation_config.top_p = 0.9` in the resolved config.

### 7c. `max_steps` counts executed tool rounds

In `_execute_trajectory` the loop runs while there are calls and
`iteration_num < max_steps` (rollout_factory.py:2311). It breaks after
`max_steps` rounds (2415-2417). Then, because the last message is a folded
tool result, it always generates one more turn (2491-2521). So an episode with
k calls (one per turn) reaches its answer turn with `max_steps = k`; it does
not need k + 1. With `max_steps = k - 1` the k-th call is written but not run,
and there is no answer turn (tests/test_trajectory.py,
`test_max_steps_k_runs_k_calls_and_still_answers`). For episodes up to 4
calls, 4 is enough. SFT replay, GRPO and the runner now all default to 4
(`DEFAULT_MAX_TOOL_STEPS`). The oracle needs at most 3.

### 7d. Stop the turn after a complete tool call (runner)

`trajectory.ends_with_tool_call(text)` is True when the text ends with one
complete JSON list of calls, followed only by whitespace or a closing code
fence. The runner's engine (`runner.make_engine`) subclasses AgentTune's
`TransformersRolloutEngine`. Its `generate()` calls `self.model.generate(...)`
with fixed arguments (transformers_engine.py:520-527) and takes no stopping
criteria, so for each turn we wrap `model.generate` on the instance, add a
`StoppingCriteria`, and remove the wrapper after the turn. The engine decodes
only the tokens up to the stop (transformers_engine.py:529-545). The loop
stores that text, stripped, as the assistant turn (rollout_factory.py:2133-2136,
2437-2450). So the turn kept in the context ends at the call, and the
template closes it as in the SFT data. tests/test_tool_call_stop.py checks
this with a scripted tiny model: with the stop, the fake text is not in the
second prompt; without it (`--no-stop-after-tool-call`), it is.

Each trace file gets `<name>.run.json` with the harness settings and the
count of turns the rule stopped.

Checks on existing files:

- `distilled_test_id.jsonl` and `distilled_test_ood.jsonl` (the SFT model
  trained on all 400 tasks): 0 of 220 traces have any model text between a
  tool call and its tool output (532 executed calls checked). For that model
  the rule is a no-op on the text.
- `zero-shot_test_id.jsonl`: 120 of 120 traces have text there;
  `zero-shot_test_ood.jsonl`: 100 of 100. Examples: `"\nTool result: 1\nANSWER: 1"`.
- SmolLM2-135M with the sft60 adapter, 3 test_id tasks, real
  `model.generate`: 6 of 9 turns stopped by the rule (the 6 call turns).
  Same raw text, same steps, same scores as with `--no-stop-after-tool-call`.
  `completion_tokens` is 2 lower per episode: the end-of-turn token after each
  call is no longer generated.

Tiny Aya tokenises `"}}]\n"` as `"}}"` then `"]\n"`, so the rule fires on the
`"]\n"` token. Its vocabulary also has 87 tokens where `]` is followed by other
text (for example `"]Bucket"`, `"}}]{"`). If the model emits one of these right
after a call, the text does not end with the list and the rule does not fire
for that turn.

GRPO does not use this engine. In the trainer path AgentTune calls
`unwrapped_model.generate(**generate_kwargs)` itself (rollout_factory.py:2047-2061)
with `generation_config=trainer.generation_config` and no stopping criteria.
The same per-call wrapper around `trainer.model`'s `generate` inside our
rollout function would work there, and the turn-context fix already handles
a turn without an end token (train_grpo.py, `with_chat_turn_context`). It is
not added: SFT teaches the model to end the turn after the list, and the
distilled traces above show no text after a call.
