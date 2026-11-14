"""Run a Hugging Face model (plus optional LoRA adapters) over tasks and write traces.

The tool loop is AgentTune's own (`create_rollout_fn` with a
`TransformersRolloutEngine`), the same loop GRPO uses during training. So the
prompt rendering, the tool-call parser and the tool-result format are the same
at training time and at evaluation time.

Example:
    python -m frugal_analyst.runner --model CohereLabs/tiny-aya-global \
        --tasks data/tasks/test_id.jsonl --out outputs/traces/zero-shot_test_id.jsonl \
        --label zero-shot

`--adapter` takes a comma-separated list. Adapters are merged into the base
weights in order. A GRPO adapter trained on top of the SFT adapter needs both:
`--adapter outputs/sft,outputs/grpo` (see scripts/train_grpo.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from .env import TOOLS
from .tasks import Task, load_tasks
from .traces import build_trace, write_traces
from .trajectory import build_prompt_messages, calls_from_trajectory

# One generation per tool round plus the final answer. The efficient path
# needs 2 or 3 calls, so 5 leaves room for a retry without letting a looping
# model run forever.
DEFAULT_MAX_TOOL_STEPS = 5
DEFAULT_MAX_NEW_TOKENS = 256  # per model turn, not per episode

# Projection modules of Cohere2 (Tiny Aya). peft has no default LoRA targets for
# model_type "cohere2" (AgentTune's ensure_lora_targets would fall back to
# "all-linear"). Llama-style models such as SmolLM2 use the same names.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Model loading (shared with the training scripts)
# ---------------------------------------------------------------------------


def pick_device(device: str | None = None) -> str:
    """cuda, then mps, then cpu. `FRUGAL_DEVICE` or `device` overrides it."""
    import torch

    device = device or os.environ.get("FRUGAL_DEVICE")
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, dtype: str | None = None):
    """bf16 where the hardware has it, fp16 on older GPUs, fp32 on CPU."""
    import torch

    if dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16  # fp32 Tiny Aya (13 GB) does not fit a 16 GB Mac
    return torch.float32


def load_tokenizer(model_id: str):
    """Tokenizer with two fixes that keep train and inference token-identical.

    1. Tiny Aya's chat template already writes `<BOS_TOKEN>`. Calling the
       tokenizer on that text adds a second BOS, and both TRL's SFT tokenizer
       and AgentTune's GRPO rollout (rollout_factory.py, `tokenizer(prompt_text)`)
       do that. Setting `add_bos_token = False` keeps one BOS everywhere.
       Templates that do not write a BOS keep the tokenizer default.
    2. A pad token, needed for batched training.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.bos_token and tok.chat_template and "bos_token" in str(tok.chat_template):
        tok.add_bos_token = False
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def stop_token_ids(model, tok) -> list[int]:
    """End-of-turn ids. Tiny Aya's generation_config stops on <EOS_TOKEN> (3),
    but a chat turn ends with <|END_OF_TURN_TOKEN|> (tokenizer.eos_token, 6)."""
    ids = []
    cfg_eos = getattr(model.generation_config, "eos_token_id", None)
    for value in [tok.eos_token_id] + (cfg_eos if isinstance(cfg_eos, list) else [cfg_eos]):
        if value is not None and value not in ids:
            ids.append(int(value))
    return ids


def split_adapters(adapter: str | list[str] | None) -> list[str]:
    if not adapter:
        return []
    if isinstance(adapter, str):
        return [a.strip() for a in adapter.split(",") if a.strip()]
    return list(adapter)


def load_model(model_id: str, adapters: list[str] | None = None, device: str = "cpu", dtype=None):
    """Base model with each LoRA adapter merged in, in order, ready for inference or training."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype or pick_dtype(device))
    model.to(device)
    for path in adapters or []:
        from peft import PeftModel

        # merge_and_unload folds the low-rank update into the weights, so the
        # next adapter (or a fresh GRPO LoRA) sits on top of this policy.
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    return model


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def build_rollout_fn(model, tok, max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS):
    """AgentTune's tool loop over a transformers model. Defaults for the tool
    prompt and tool-result format are AgentTune's, the same as in GRPO."""
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    model.generation_config.eos_token_id = stop_token_ids(model, tok)
    engine = TransformersRolloutEngine(model, tok)
    return create_rollout_fn(rollout_engine=engine, tools=list(TOOLS), max_steps=max_tool_steps)


def run_episode(rollout_fn, tok, task: Task, gen_kwargs: dict) -> dict[str, Any]:
    """One task through the tool loop. Returns calls, final text and raw stats."""
    started = time.perf_counter()
    out = rollout_fn([build_prompt_messages(task)], None, **gen_kwargs)
    latency = time.perf_counter() - started
    trajectory = out["trajectories"][0]
    completion_ids = out["completion_ids"][0]
    env_mask = out["env_mask"][0]
    calls, final_text = calls_from_trajectory(trajectory)
    return {
        "calls": calls,
        "final_text": final_text,
        # Same string the GRPO reward function receives during training.
        "raw_completion": tok.decode(completion_ids, skip_special_tokens=True),
        # Model-generated tokens only; env_mask is 0 on tool-output tokens.
        "completion_tokens": int(sum(env_mask)) if env_mask else len(completion_ids),
        "latency_s": round(latency, 3),
        "trajectory": trajectory,
    }


def run_model(
    model_id: str,
    adapter: str | list[str] | None,
    tasks: list[Task],
    out_path: str | Path,
    label: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS,
    temperature: float = 0.0,
    seed: int = 0,
    limit: int | None = None,
    device: str | None = None,
    dtype: str | None = None,
    db_dir: str | Path | None = None,
) -> list[dict]:
    """Run the model over `tasks`, write one trace per task to `out_path`, return the traces.

    Episodes run one at a time: AgentTune's loop handles one conversation per
    generate() call, and its threaded mode is not safe on one shared model.
    """
    import torch
    from transformers import set_seed

    set_seed(seed)
    device = pick_device(device)
    torch_dtype = pick_dtype(device, dtype)
    tasks = tasks[:limit] if limit else tasks
    adapters = split_adapters(adapter)
    print(f"[runner] model={model_id} adapters={adapters} device={device} dtype={torch_dtype} tasks={len(tasks)}")

    tok = load_tokenizer(model_id)
    model = load_model(model_id, adapters, device, torch_dtype)
    model.eval()
    rollout_fn = build_rollout_fn(model, tok, max_tool_steps)

    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False, temperature=None)  # greedy

    traces: list[dict] = []
    failures = 0
    for n, task in enumerate(tasks, 1):
        try:
            ep = run_episode(rollout_fn, tok, task, gen_kwargs)
            trace = build_trace(
                task, ep["calls"], ep["final_text"], label,
                raw_completion=ep["raw_completion"], completion_tokens=ep["completion_tokens"],
                latency_s=ep["latency_s"], db_dir=db_dir,
            )
        except Exception as exc:  # one bad episode must not lose a long eval run
            failures += 1
            print(f"[runner] task {task.id} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            trace = build_trace(task, [], None, label, raw_completion=None, db_dir=db_dir)
        traces.append(trace)
        print(
            f"[runner] {n}/{len(tasks)} {task.id} correct={trace['correct']} reward={trace['reward']:.3f} "
            f"calls={trace['cost']['tool_calls']} latency={trace['latency_s']}s"
        )
        if n % 10 == 0:
            write_traces(out_path, traces)  # partial results survive a crash
    write_traces(out_path, traces)
    if traces:
        acc = sum(t["correct"] for t in traces) / len(traces)
        mean_reward = sum(t["reward"] for t in traces) / len(traces)
        print(f"[runner] wrote {len(traces)} traces to {out_path}: accuracy={acc:.3f} mean_reward={mean_reward:.3f} failures={failures}")
    return traces


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir(s) or hub id(s), comma-separated, merged in order")
    p.add_argument("--tasks", required=True, help="task jsonl, e.g. data/tasks/test_id.jsonl")
    p.add_argument("--out", required=True, help="trace jsonl to write")
    p.add_argument("--label", required=True, help="model label: zero-shot, distilled, grpo, grpo-ungated")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="per model turn")
    p.add_argument("--max-tool-steps", type=int, default=DEFAULT_MAX_TOOL_STEPS)
    p.add_argument("--temperature", type=float, default=0.0, help="0 means greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    p.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    args = p.parse_args(argv)
    run_model(
        args.model, args.adapter, load_tasks(args.tasks), args.out, args.label,
        max_new_tokens=args.max_new_tokens, max_tool_steps=args.max_tool_steps,
        temperature=args.temperature, seed=args.seed, limit=args.limit,
        device=args.device, dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
