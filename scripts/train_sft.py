"""Distillation warm start: LoRA SFT on the oracle teacher traces, with AgentTune.

How the teacher text gets the exact rollout format
--------------------------------------------------
GRPO and the runner both drive the model through AgentTune's tool loop
(rollout_factory._execute_trajectory). For Tiny Aya that loop adds the tool
list to the system turn, parses a `[{"tool_name": ..., "parameters": ...}]`
reply, and folds each tool result into a user turn. To train on exactly those
prompts we replay every teacher trace through the same loop:

1. `ScriptedTeacherEngine` is an AgentTune `RolloutEngine` that "generates" the
   teacher's next turn instead of sampling. AgentTune's `Project.collect_rollout`
   drives it with the real tools, so every tool result is produced by the loop.
2. The engine records each conversation it was asked to continue. Each one
   becomes one SFT row: prompt = that conversation rendered with
   `_render_chat_template` (the renderer the GRPO rollout uses), completion =
   the teacher turn plus the template's end-of-turn tokens.
3. `Project.distill` runs the student trainer through `ExactFormatSFTFactory`
   (TRL SFTTrainer + LoRA, loss on completions only).

AgentTune's generic distill rows (`EventLog.as_dataset_rows`) write calls as
`<tool_call>{"name": ...}</tool_call>` and leave out the tool list, which is
not what Tiny Aya sees in the rollout. That is why the factory trains on the
recorded rows. See docs/training-notes.md.

Example (smoke test on CPU):
    python scripts/train_sft.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --limit 8 --max-steps 3 --out outputs/smoke/sft --device cpu

For the SFT-then-GRPO pipeline, train on the SFT subset so GRPO gets templates
SFT never saw: `--teacher data/teacher/train_sft.jsonl`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from frugal_analyst.env import TOOLS  # noqa: E402
from frugal_analyst.reward import RewardConfig, ToolCall, score_trajectory  # noqa: E402
from frugal_analyst.runner import (  # noqa: E402
    LORA_TARGET_MODULES,
    DEFAULT_MAX_TOOL_STEPS,
    load_tokenizer,
    pick_device,
    pick_dtype,
)
from frugal_analyst.traces import read_traces  # noqa: E402
from frugal_analyst.trajectory import (  # noqa: E402
    build_prompt_messages,
    calls_from_trajectory,
    format_tool_call_turn,
    load_task_index,
)


# ---------------------------------------------------------------------------
# Teacher replay
# ---------------------------------------------------------------------------


def teacher_turns(trace: dict) -> list[str]:
    """The teacher's assistant turns: one JSON call list per tool call, then the answer.

    The oracle makes one call per turn, which is also what the efficient path
    looks like at inference.
    """
    turns = []
    for step in trace["steps"]:
        if step["type"] == "tool_call":
            turns.append(format_tool_call_turn([ToolCall(step["name"], step["arguments"])]))
        elif step["type"] == "final":
            turns.append(step["text"])
    return turns


def _first_user_text(conversation: list[dict]) -> str:
    for msg in conversation:
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def make_teacher_engine(tokenizer, scripts: dict[str, list[str]]):
    """An AgentTune RolloutEngine that replays teacher turns. Built in a function
    so AgentTune is imported only when the script runs."""
    from agenttune.agentic.rollout_engines.base import RolloutEngine

    class ScriptedTeacherEngine(RolloutEngine):
        def __init__(self):
            self.tokenizer = tokenizer
            self.calls: list[tuple[list[dict], list[dict], str]] = []

        def _get_tokenizer(self):  # used by _execute_trajectory when there is no trainer
            return self.tokenizer

        def generate(self, prompts, tools, gen_cfg):
            conversation = prompts
            script = scripts[_first_user_text(conversation)]
            turn = sum(1 for m in conversation if m.get("role") == "assistant")
            text = script[turn] if turn < len(script) else ""
            self.calls.append((copy.deepcopy(conversation), copy.deepcopy(tools), text))
            return {"completions": [text], "completions_raw": [text], "logprobs": [None], "metadata": {}}

    return ScriptedTeacherEngine()


def render_rows(tokenizer, recorded) -> list[dict]:
    """Prompt/completion rows from the recorded conversations.

    The prompt uses AgentTune's `_render_chat_template`, the same call the GRPO
    rollout makes (rollout_factory.py, `_gen` path A). The completion is the
    teacher text plus whatever the template writes after an assistant turn, cut
    after the end-of-turn token, because generation stops there.
    """
    from agenttune.agentic.rollout_engines.rollout_factory import _render_chat_template

    eos = tokenizer.eos_token
    rows = []
    for conversation, schemas, text in recorded:
        prompt = _render_chat_template(tokenizer, conversation, schemas, enable_thinking=False, tokenize=False)
        full = _render_chat_template(
            tokenizer, conversation + [{"role": "assistant", "content": text}], schemas,
            enable_thinking=False, tokenize=False, add_generation_prompt=False,
        )
        completion = full[len(prompt):] if full.startswith(prompt) else text + eos
        cut = completion.find(eos, len(text)) if eos else -1
        if cut >= 0:
            completion = completion[: cut + len(eos)]
        rows.append({"prompt": prompt, "completion": completion})
    return rows


def check_engine_render_matches(tokenizer, recorded) -> bool:
    """The runner renders through TransformersRolloutEngine (tokenizer.apply_chat_template
    with tools=...), GRPO through _render_chat_template. Check they agree on one prompt."""
    from agenttune.agentic.rollout_engines.rollout_factory import _render_chat_template

    conversation, schemas, _ = recorded[-1]
    a = _render_chat_template(tokenizer, conversation, schemas, enable_thinking=False, tokenize=False)
    b = tokenizer.apply_chat_template(conversation, tools=schemas or None, add_generation_prompt=True, tokenize=False)
    return a == b


# ---------------------------------------------------------------------------
# Student trainer factory for Project.distill
# ---------------------------------------------------------------------------


class ExactFormatSFTFactory:
    """`trainer_factory` for AgentTune's Project.distill: TRL SFTTrainer + LoRA on
    the recorded rollout-format rows (loss on the completion tokens only)."""

    def __init__(self, rows, tokenizer, lora_r: int, device: str, dtype, gradient_checkpointing: bool):
        self.rows = rows
        self.tokenizer = tokenizer
        self.lora_r = lora_r
        self.device = device
        self.dtype = dtype
        self.gradient_checkpointing = gradient_checkpointing
        self.trainer = None

    def __call__(self, *, model, train_dataset, **kwargs):
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM
        from trl import SFTConfig, SFTTrainer

        print(f"[sft] Project.distill handed over {len(train_dataset)} generic rows; training on {len(self.rows)} exact-format rows")
        student = AutoModelForCausalLM.from_pretrained(model, dtype=self.dtype)
        cfg = SFTConfig(
            **kwargs,
            bf16=self.dtype == torch.bfloat16 and self.device.startswith("cuda"),
            fp16=self.dtype == torch.float16 and self.device.startswith("cuda"),
            use_cpu=self.device == "cpu",
            gradient_checkpointing=self.gradient_checkpointing,
        )
        peft_config = LoraConfig(
            r=self.lora_r, lora_alpha=2 * self.lora_r, lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
        )
        self.trainer = SFTTrainer(
            model=student, args=cfg, train_dataset=Dataset.from_list(self.rows),
            processing_class=self.tokenizer, peft_config=peft_config,
        )
        return self.trainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    p.add_argument("--teacher", default=str(REPO_ROOT / "data/teacher/train.jsonl"),
                   help="teacher traces; data/teacher/train_sft.jsonl leaves the RL-only templates for GRPO")
    p.add_argument("--out", default=str(REPO_ROOT / "outputs/sft"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help="optimizer steps; -1 means use --epochs")
    p.add_argument("--limit", type=int, default=None, help="use only the first N teacher traces")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=3072, help="rows longer than this are dropped, not cut")
    p.add_argument("--max-tool-steps", type=int, default=DEFAULT_MAX_TOOL_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--push-to-hub", default=None, metavar="REPO", help="upload the adapter to this HF model repo (off by default)")
    args = p.parse_args(argv)

    from transformers import set_seed

    from agenttune.agentic.project import Project

    set_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.model)
    tasks = load_task_index()
    traces = read_traces(args.teacher)
    traces = traces[: args.limit] if args.limit else traces
    prompts, scripts, by_question = [], {}, {}
    for trace in traces:
        task = tasks[trace["task_id"]]
        messages = build_prompt_messages(task)
        scripts[messages[-1]["content"]] = teacher_turns(trace)
        by_question[messages[-1]["content"]] = (task, trace)
        prompts.append(messages)
    print(f"[sft] {len(traces)} teacher traces, device={device}, dtype={dtype}")

    # 1. Replay the teacher through AgentTune's real tool loop.
    engine = make_teacher_engine(tok, scripts)
    project = Project()
    project.collect_rollout(engine, prompts, tools=list(TOOLS), max_steps=args.max_tool_steps)

    # Check the replay: the loop must execute the teacher's calls and reach its answer.
    replay_ok = 0
    for traj in project._native_trajectories:
        task, trace = by_question[_first_user_text(traj.task)]
        calls, final = calls_from_trajectory(traj)
        score = score_trajectory(task, calls, final, RewardConfig())
        replay_ok += int(abs(score.reward - trace["reward"]) < 1e-9 and score.correct)
    print(f"[sft] replay check: {replay_ok}/{len(project._native_trajectories)} episodes match the teacher score")
    if replay_ok != len(project._native_trajectories):
        raise SystemExit("[sft] teacher replay does not match the teacher traces; the format would be wrong")

    # 2. Rows in the exact rollout format.
    rows = render_rows(tok, engine.calls)
    same_render = check_engine_render_matches(tok, engine.calls)
    print(f"[sft] trainer render == inference engine render: {same_render}")
    lengths = [len(tok(r["prompt"] + r["completion"])["input_ids"]) for r in rows]
    kept = [r for r, n in zip(rows, lengths, strict=True) if n <= args.max_length]
    print(f"[sft] {len(rows)} rows, longest {max(lengths)} tokens, {len(rows) - len(kept)} dropped for --max-length {args.max_length}")
    with (out / "sft_rows_preview.jsonl").open("w", encoding="utf-8") as f:
        for row in kept[:4]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3. Distill with AgentTune's Project.distill.
    factory = ExactFormatSFTFactory(kept, tok, args.lora_r, device, dtype, not args.no_gradient_checkpointing)
    project.distill(
        args.model,
        trainer_factory=factory,
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * max(args.max_steps, 1))) if args.max_steps > 0 else 5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=args.max_length,
        seed=args.seed,
    )

    factory.trainer.save_model(str(out))  # LoRA adapter only
    tok.save_pretrained(str(out))
    history = factory.trainer.state.log_history
    meta = {
        "base_model": args.model,
        "args": vars(args),
        "n_teacher_traces": len(traces),
        "n_rows": len(kept),
        "replay_ok": replay_ok,
        "trainer_render_matches_engine": same_render,
        "train_loss_first": next((h["loss"] for h in history if "loss" in h), None),
        "train_loss_last": next((h["loss"] for h in reversed(history) if "loss" in h), None),
        "global_step": factory.trainer.state.global_step,
    }
    (out / "frugal_sft_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    with (out / "log_history.jsonl").open("w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")
    print(f"[sft] saved adapter to {out}: {json.dumps({k: meta[k] for k in ('n_rows', 'global_step', 'train_loss_first', 'train_loss_last')})}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push_to_hub)
        print(f"[sft] pushed to {args.push_to_hub}")


if __name__ == "__main__":
    main()
