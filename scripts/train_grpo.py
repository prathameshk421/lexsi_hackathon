"""GRPO with tools, on top of the distilled LoRA adapter, with AgentTune.

The trainer is `create_agentic_trainer("grpo", ...)` (AgentTune's TrlAgenticGrpo
around TRL's GRPOTrainer). The tool loop is AgentTune's `create_rollout_fn`,
the same loop the runner uses for evaluation. Our rollout function wraps it to
fix three things we found in the source (details in docs/training-notes.md):

1. Per-turn length and temperature. In the trainer path AgentTune generates
   with `max_new_tokens=getattr(trainer, "_max_new_tokens", 256)` and
   `temperature=getattr(trainer, "_temperature", 0.7)`, and these explicit
   values override GRPOConfig. We set both attributes on the trainer.
2. Eval mode while sampling. The trainer is in train mode during rollouts, so
   gradient checkpointing turns the KV cache off and LoRA dropout is active.
   We switch to eval mode for generation and back afterwards.
3. Turn context. AgentTune puts the raw tool output tokens between two model
   turns in `completion_ids`, but the model saw a chat-formatted user turn
   there ("Tool result for run_sql: ..."). We replace those tokens with the
   exact template text, so the policy gradient for turn 2 and later is computed
   in the same context the tokens were sampled in. Tool tokens stay masked out
   of the loss (env_mask = 0). Turn it off with --no-turn-context-fix.

No-signal guard (--stop-if-no-signal N, default 3): if no GRPO group has any
reward spread in each of the first N optimizer steps, every advantage is 0 and
training cannot learn. The script then stops, keeps reward_log.jsonl and
trajectories.jsonl, saves no adapter, and exits with code 4.

Tasks: after SFT on data/teacher/train_sft.jsonl, train GRPO on
data/tasks/train_rl.jsonl (RL-only templates plus forgetting checks).

Starting point (--init-adapter): the SFT adapter is merged into the base
weights, and GRPO trains a fresh LoRA on top. TRL computes the KL reference by
disabling the trainable adapter, so the reference is the distilled policy, not
the zero-shot base. At inference merge both, in order:
    python -m frugal_analyst.runner --adapter outputs/sft,outputs/grpo ...

Example (smoke test on CPU):
    python scripts/train_grpo.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --init-adapter outputs/smoke/sft --limit 4 --max-steps 1 --num-generations 2 \
        --batch-size 2 --grad-accum 1 --max-completion-length 48 --max-tool-steps 2 \
        --out outputs/smoke/grpo --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from frugal_analyst.env import TOOLS  # noqa: E402
from frugal_analyst.runner import (  # noqa: E402
    DEFAULT_MAX_TOOL_STEPS,
    LORA_TARGET_MODULES,
    load_model,
    load_tokenizer,
    pick_device,
    pick_dtype,
    split_adapters,
)
from frugal_analyst.tasks import load_tasks  # noqa: E402
from frugal_analyst.trajectory import build_prompt_messages, make_grpo_reward  # noqa: E402


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------


def _mask_runs(mask: list[int]) -> list[tuple[int, int, int]]:
    """(value, start, end) for each run of equal values in a 0/1 mask."""
    runs = []
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            runs.append((mask[start], start, i))
            start = i
    return runs


def with_chat_turn_context(out: dict, tokenizer, schemas: list[dict]) -> tuple[dict, int]:
    """Replace raw tool-output tokens with the chat-template text the model really saw.

    Returns the new rollout output and how many trajectories were rebuilt.
    A trajectory whose turns cannot be matched one to one is left unchanged.
    """
    from agenttune.agentic.rollout_engines.rollout_factory import _render_chat_template

    def render(conv, gen):
        return _render_chat_template(
            tokenizer, conv, schemas, enable_thinking=False, tokenize=False, add_generation_prompt=gen
        )

    eos_id = tokenizer.eos_token_id
    eos = tokenizer.eos_token
    rebuilt = 0
    new_ids, new_masks, new_logprobs = [], [], []
    for i, traj in enumerate(out["trajectories"]):
        ids, mask = out["completion_ids"][i], out["env_mask"][i]
        conv = (traj.metadata or {}).get("conversation") or []
        turns = [k for k, m in enumerate(conv) if m.get("role") == "assistant"]
        model_runs = [ids[s:e] for v, s, e in _mask_runs(mask or []) if v == 1]
        if not mask or not turns or len(model_runs) != len(turns):
            new_ids.append(ids), new_masks.append(mask), new_logprobs.append(out["logprobs"][i])
            continue
        try:
            seq, seq_mask = [], []
            for k, a in enumerate(turns):
                seq += model_runs[k]
                seq_mask += [1] * len(model_runs[k])
                if k + 1 == len(turns):
                    break
                closed = render(conv[: a + 1], gen=False)
                nxt = render(conv[: turns[k + 1]], gen=True)
                if not nxt.startswith(closed):
                    raise ValueError("chat template is not prefix-preserving")
                # `closed` ends with the template's text after the assistant
                # content (Tiny Aya: <|END_RESPONSE|><|END_OF_TURN_TOKEN|>;
                # ChatML: <|im_end|>\n). The model emitted that text only up to
                # its eos token, so the rest belongs to the glue. A turn that hit
                # the token limit emitted none of it.
                before = render(conv[:a], gen=True)
                content = str(conv[a].get("content") or "")
                tail = closed[len(before):]
                if not (closed.startswith(before) and tail.startswith(content)):
                    raise ValueError("cannot locate the assistant turn in the template")
                closing = tail[len(content):]
                if model_runs[k] and model_runs[k][-1] == eos_id and eos and eos in closing:
                    closing = closing[closing.index(eos) + len(eos):]
                glue = closing + nxt[len(closed):]
                glue_ids = tokenizer.encode(glue, add_special_tokens=False)
                seq += glue_ids
                seq_mask += [0] * len(glue_ids)
        except Exception:
            new_ids.append(ids), new_masks.append(mask), new_logprobs.append(out["logprobs"][i])
            continue
        rebuilt += 1
        new_ids.append(seq)
        new_masks.append(seq_mask)
        # Transformers path: AgentTune already returns 0.0 logprobs (TRL recomputes them).
        new_logprobs.append([0.0] * len(seq))
    out = dict(out)
    out["completion_ids"], out["env_mask"], out["logprobs"] = new_ids, new_masks, new_logprobs
    return out, rebuilt


def check_turn_context(out: dict, tokenizer, schemas: list[dict]) -> str:
    """How many rebuilt sequences decode to exactly the chat the model saw.

    Text after a model turn's end token is not compared (a final turn that hit
    the token limit has no end token)."""
    from agenttune.agentic.rollout_engines.rollout_factory import _render_chat_template

    exact = 0
    for i, traj in enumerate(out["trajectories"]):
        conv = (traj.metadata or {}).get("conversation") or []
        if not conv:
            continue
        seen = tokenizer.decode(out["prompt_ids"][i] + out["completion_ids"][i], skip_special_tokens=False)
        chat = _render_chat_template(
            tokenizer, conv, schemas, enable_thinking=False, tokenize=False, add_generation_prompt=False
        )
        exact += int(chat.startswith(seen) or seen.startswith(chat))
    return f"{exact}/{len(out['trajectories'])} sequences match the rendered chat"


def make_rollout_func(max_tool_steps, max_new_tokens, temperature, trajectories_path, turn_context_fix=True):
    """AgentTune's create_rollout_fn plus the three fixes in the module docstring."""
    from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema, create_rollout_fn
    from agenttune.agentic.trajectory.dataset import trajectory_writer

    base = create_rollout_fn(
        tools=list(TOOLS),
        max_steps=max_tool_steps,
        on_trajectory_end=trajectory_writer(trajectories_path) if trajectories_path else None,
    )
    schemas = [_build_tool_schema(t) for t in TOOLS]
    stats = {"calls": 0, "rebuilt": 0, "trajectories": 0}

    def frugal_rollout(prompts, trainer):
        trainer._max_new_tokens = max_new_tokens
        trainer._temperature = temperature
        model = trainer.model
        was_training = model.training
        model.eval()
        try:
            out = base(prompts, trainer)
        finally:
            if was_training:
                model.train()
        if turn_context_fix:
            tok = trainer.processing_class
            tok = getattr(tok, "tokenizer", tok)
            out, rebuilt = with_chat_turn_context(out, tok, schemas)
            stats["rebuilt"] += rebuilt
            if stats["calls"] == 0:
                print(f"[grpo] turn-context fix: {check_turn_context(out, tok, schemas)}")
        stats["calls"] += 1
        stats["trajectories"] += len(out["trajectories"])
        return out

    frugal_rollout.stats = stats
    return frugal_rollout


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def with_first_batch_print(reward_fn, max_chars: int = 2000):
    """Print what the reward function receives on its first call: one raw
    completion, the calls parsed from the text and from the trajectory, and the reward."""
    from frugal_analyst.trajectory import calls_from_trajectory, completion_text, parse_completion

    state = {"done": False}

    def frugal_reward(completions, **kwargs):
        rewards = reward_fn(completions, **kwargs)
        if not state["done"]:
            state["done"] = True
            text = completion_text(completions[0])
            trajs = kwargs.get("trajectories") or [None]
            print("[grpo] ---- first completion as the reward function receives it ----")
            print(f"[grpo] type(completions[0]) = {type(completions[0]).__name__}; kwargs = {sorted(kwargs)}")
            print(f"[grpo] task_id = {kwargs.get('task_id', [None])[0]}")
            print(text[:max_chars])
            print(f"[grpo] parse_completion(text) -> {parse_completion(text)}")
            if trajs[0] is not None:
                print(f"[grpo] calls_from_trajectory -> {calls_from_trajectory(trajs[0])}")
            print(f"[grpo] rewards for this batch = {rewards}")
            print("[grpo] -----------------------------------------------------------")
        return rewards

    return frugal_reward


def with_group_spread_metric(reward_fn, num_generations: int):
    """Log `frugal/frac_groups_no_spread`: the share of GRPO groups whose rewards are all equal.

    A group with equal rewards has advantage 0, so it gives no gradient. TRL
    logs `frac_reward_zero_std` too, but with `scale_rewards="batch"` TRL
    computes it from the batch std, not the group std (trl grpo_trainer.py:2345-2359).
    This metric is per group for every setting. TRL repeats each prompt
    `num_generations` times in a row (RepeatSampler), so a group is a run of
    `num_generations` rewards.
    """

    def frugal_reward(completions, **kwargs):
        rewards = reward_fn(completions, **kwargs)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and rewards and len(rewards) % num_generations == 0:
            groups = [rewards[i:i + num_generations] for i in range(0, len(rewards), num_generations)]
            flat = sum(1 for g in groups if max(g) - min(g) <= 1e-6)
            log_metric("frugal/frac_groups_no_spread", flat / len(groups))
        return rewards

    return frugal_reward


# Exit code when the no-signal guard stops training. 1 is a Python error, 2 is argparse.
EXIT_NO_SIGNAL = 4
# Our per-group metric first; TRL's metric if ours is missing.
NO_SIGNAL_KEYS = ("frugal/frac_groups_no_spread", "frac_reward_zero_std")


class NoSignalGuard:
    """Stop GRPO when the first N optimizer steps have no reward spread in any group.

    Then every advantage is 0 and the steps only burn GPU time. Our last real
    run did that: reward 1.0 on every rollout and gradient 0. N <= 0 turns the
    guard off. It looks only at the first N steps, so a later quiet step never
    stops a run that has learned something.
    """

    def __init__(self, n_steps: int):
        self.n_steps = n_steps
        self.values: list[float] = []
        self.stopped = False

    def update(self, logs: dict) -> bool:
        """Read one log line. Returns True when training must stop now."""
        if self.n_steps <= 0 or len(self.values) >= self.n_steps:
            return False
        value = next((logs[k] for k in NO_SIGNAL_KEYS if logs.get(k) is not None), None)
        if value is None:
            return False  # not a training-step line (for example the final summary)
        self.values.append(float(value))
        if len(self.values) == self.n_steps and all(v >= 1.0 - 1e-6 for v in self.values):
            self.stopped = True
        return self.stopped


def make_reward_logger(path: Path, guard: NoSignalGuard | None = None):
    """A TrainerCallback that appends every logged metrics dict (mean reward per step) to a jsonl
    file, and stops training when `guard` says there is no learning signal."""
    from transformers import TrainerCallback

    class RewardLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            if state.is_world_process_zero and any("reward" in k for k in logs):
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": state.global_step, **logs}, default=float) + "\n")
            if guard is not None and guard.update(logs):
                print(f"[grpo] STOP: no reward spread in any group for the first {guard.n_steps} steps "
                      f"(frac_groups_no_spread = {guard.values}). Every advantage is 0, so training "
                      f"cannot learn. Use harder tasks (data/tasks/train_rl.jsonl), a higher temperature, "
                      f"or turn this off with --stop-if-no-signal 0.")
                control.should_training_stop = True

    return RewardLogger()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    p.add_argument("--init-adapter", default=None, help="SFT adapter dir or hub id, merged into the base first (comma list allowed)")
    p.add_argument("--tasks", default=str(REPO_ROOT / "data/tasks/train.jsonl"),
                   help="task jsonl; use data/tasks/train_rl.jsonl after SFT on train_sft")
    p.add_argument("--out", default=str(REPO_ROOT / "outputs/grpo"))
    p.add_argument("--max-steps", type=int, default=40, help="optimizer steps")
    p.add_argument("--num-generations", type=int, default=4, help="completions per prompt (the GRPO group)")
    p.add_argument("--batch-size", type=int, default=4, help="per-device completions per micro-batch")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04, help="KL weight against the starting (distilled) policy")
    p.add_argument("--temperature", type=float, default=0.8)
    # Each of these is a GRPOConfig field in trl 1.7.1. top_p reaches the rollout
    # through trainer.generation_config (see docs/training-notes.md, section 2e).
    p.add_argument("--top-p", type=float, default=0.95, help="sampling top-p (0.95 is AgentTune's default)")
    p.add_argument("--scale-rewards", default="group", choices=["group", "batch", "none"],
                   help="divide advantages by the group std, the batch std, or nothing")
    p.add_argument("--loss-type", default="dapo",
                   choices=["grpo", "dapo", "bnpo", "dr_grpo", "cispo", "sapo", "luspo", "vespo"])
    p.add_argument("--epsilon-high", type=float, default=None,
                   help="upper clip bound; default: same as epsilon (0.2). DAPO uses 0.28")
    p.add_argument("--stop-if-no-signal", type=int, default=3, metavar="N",
                   help=f"stop (exit code {EXIT_NO_SIGNAL}) if the first N steps have no reward spread "
                        "in any group; 0 turns it off")
    p.add_argument("--max-completion-length", type=int, default=256, help="max new tokens PER MODEL TURN")
    p.add_argument("--max-tool-steps", type=int, default=DEFAULT_MAX_TOOL_STEPS,
                   help="tool calls executed per episode; one answer turn always follows")
    p.add_argument("--reward-mode", default="gated", choices=["gated", "ungated"])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="use only the first N training tasks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--no-turn-context-fix", action="store_true")
    p.add_argument("--push-to-hub", default=None, metavar="REPO", help="upload the adapter to this HF model repo (off by default)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed

    from agenttune.core.backend_factory import create_agentic_trainer

    set_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reward_log = out / "reward_log.jsonl"
    reward_log.unlink(missing_ok=True)
    (out / "trajectories.jsonl").unlink(missing_ok=True)

    # Dataset: chat prompt plus task_id, which TRL forwards to the reward function.
    tasks = load_tasks(args.tasks)
    tasks = tasks[: args.limit] if args.limit else tasks
    dataset = Dataset.from_list([{"prompt": build_prompt_messages(t), "task_id": t.id} for t in tasks])
    per_step = args.batch_size * args.grad_accum
    if per_step % args.num_generations:
        raise SystemExit("--batch-size * --grad-accum must be divisible by --num-generations")
    print(f"[grpo] {len(tasks)} tasks, {per_step // args.num_generations} prompts x {args.num_generations} generations per step, device={device}, dtype={dtype}")

    tok = load_tokenizer(args.model)
    init_adapters = split_adapters(args.init_adapter)
    model = load_model(args.model, init_adapters, device, dtype)
    rollout_func = make_rollout_func(
        args.max_tool_steps, args.max_completion_length, args.temperature,
        out / "trajectories.jsonl", turn_context_fix=not args.no_turn_context_fix,
    )
    expected = {
        "num_generations": args.num_generations,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "beta": args.beta,
        "temperature": args.temperature,
        "max_completion_length": args.max_completion_length,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "logging_steps": 1,  # the no-signal guard reads one log line per optimizer step
        "top_p": args.top_p,
        "scale_rewards": args.scale_rewards,
        "loss_type": args.loss_type,
        "epsilon_high": args.epsilon_high,
    }
    guard = NoSignalGuard(args.stop_if_no_signal)
    reward = with_group_spread_metric(make_grpo_reward(args.reward_mode), args.num_generations)
    trainer = create_agentic_trainer(
        "grpo",
        model=model,
        processing_class=tok,
        tools=list(TOOLS),
        rollout_func=rollout_func,
        reward_funcs=[with_first_batch_print(reward)],
        train_dataset=dataset,
        output_dir=str(out),
        peft_config=LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
        ),
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        bf16=dtype == torch.bfloat16 and device.startswith("cuda"),
        fp16=dtype == torch.float16 and device.startswith("cuda"),
        use_cpu=device == "cpu",
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_vllm=False,
        callbacks=[make_reward_logger(reward_log, guard)],
        **expected,
    )

    # TrlAgenticGrpo.train() is setup_data + setup_trainer + trainer.train + save_model.
    # We run the same steps one by one so we can check the resolved config first,
    # because AgentTune drops kwargs it does not recognise without a warning.
    trainer.setup_data()
    trainer.setup_trainer()
    resolved = trainer.trainer.args
    print("[grpo] resolved GRPOConfig:")
    for key in [*expected, "steps_per_generation", "generation_batch_size", "warmup_steps", "bf16", "fp16",
                "use_cpu", "gradient_checkpointing", "use_vllm", "epsilon", "num_iterations", "remove_unused_columns"]:
        print(f"    {key} = {getattr(resolved, key, '<missing>')}")
    wrong = {k: (v, getattr(resolved, k, None)) for k, v in expected.items() if getattr(resolved, k, None) != v}
    # The rollout samples with trainer.generation_config, so top_p must be there too.
    gen_top_p = getattr(getattr(trainer.trainer, "generation_config", None), "top_p", None)
    print(f"    generation_config.top_p = {gen_top_p}")
    if gen_top_p != args.top_p:
        wrong["generation_config.top_p"] = (args.top_p, gen_top_p)
    if wrong:
        raise SystemExit(f"[grpo] settings did not take effect (wanted, got): {wrong}")
    if trainer.trainer.rollout_func is not rollout_func:
        raise SystemExit("[grpo] the AgentTune rollout function is not attached to the trainer")
    trainer.trainer.model.print_trainable_parameters()

    result = trainer.trainer.train()
    trainer._train_result = result
    meta = {
        "base_model": args.model,
        "init_adapters": init_adapters,
        "inference_adapters": init_adapters + [str(out)],
        "args": vars(args),
        "rollout_stats": rollout_func.stats,
        "global_step": result.global_step,
        "train_loss": result.training_loss,
        "stopped_no_signal": guard.stopped,
        "no_signal_values": guard.values,
    }
    if guard.stopped:
        # No adapter: it learned nothing, and a saved one could be evaluated by mistake.
        # reward_log.jsonl and trajectories.jsonl are already on disk.
        meta["inference_adapters"] = None
        (out / "frugal_grpo_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[grpo] stopped after {result.global_step} steps: no learning signal. Logs are in {out}.")
        sys.exit(EXIT_NO_SIGNAL)
    trainer.save_model(str(out))
    (out / "frugal_grpo_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[grpo] saved adapter to {out}. rollout stats: {rollout_func.stats}")
    print(f"[grpo] evaluate with: python -m frugal_analyst.runner --model {args.model} --adapter {','.join(meta['inference_adapters'])} ...")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push_to_hub)
        print(f"[grpo] pushed to {args.push_to_hub}")


if __name__ == "__main__":
    main()
