"""Optional live model backend.

The app never needs it. Replay and scripted policies work without model weights.
A backend is "available" only when all of this is true:

* `FRUGAL_LIVE_MODEL=1` is set (an explicit opt in, so a CPU Space never loads 3B weights by accident),
* `torch`, `transformers` and `agenttune` can be found,
* `frugal_analyst.runner` imports.

Optional settings: `FRUGAL_MODEL_ID` (default CohereLabs/tiny-aya-global),
`FRUGAL_ADAPTER` (comma separated LoRA dirs), `FRUGAL_MODEL_LABEL` (name in the UI).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import threading
from typing import Protocol

DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"
REQUIRED_MODULES = ("torch", "transformers", "agenttune")


class ModelBackend(Protocol):
    name: str

    def available(self) -> bool:
        """True if `run` can work on this machine right now."""

    def run(self, database: str, question: str) -> tuple[list, str | None, str | None]:
        """Answer one question. Returns (tool calls, final text, raw completion).

        A tool call is a `frugal_analyst.reward.ToolCall` or a dict with `name` and `arguments`.
        """


class NoBackend:
    name = "model"

    def available(self) -> bool:
        return False

    def run(self, database: str, question: str):
        raise RuntimeError("no model backend is available")


class RunnerBackend:
    """Runs a Hugging Face model through `frugal_analyst.runner`. Loads the model on first use."""

    def __init__(self) -> None:
        self.name = os.environ.get("FRUGAL_MODEL_LABEL", "tiny-aya")[:30]
        self._lock = threading.Lock()
        self._loaded = None  # (runner module, rollout_fn, tokenizer)

    def available(self) -> bool:
        if os.environ.get("FRUGAL_LIVE_MODEL") != "1":
            return False
        try:
            for mod in REQUIRED_MODULES:
                if importlib.util.find_spec(mod) is None:
                    return False
            importlib.import_module("frugal_analyst.runner")
        except Exception:  # missing file, missing torch, a broken import: all mean "off"
            return False
        return True

    def _load(self):
        if self._loaded is None:
            runner = importlib.import_module("frugal_analyst.runner")
            model_id = os.environ.get("FRUGAL_MODEL_ID", DEFAULT_MODEL_ID)
            adapters = runner.split_adapters(os.environ.get("FRUGAL_ADAPTER"))
            device = runner.pick_device()
            tok = runner.load_tokenizer(model_id)
            model = runner.load_model(model_id, adapters, device, runner.pick_dtype(device))
            model.eval()
            self._loaded = (runner, runner.build_rollout_fn(model, tok), tok)
        return self._loaded

    def run(self, database: str, question: str):
        from frugal_analyst.tasks import Task

        with self._lock:  # one generation at a time on one model
            runner, rollout_fn, tok = self._load()
            task = Task(
                id="live", domain=database, level=0, template="live", question=question,
                answer_type="text", gold_sql="", gold_answer="", oracle_calls=[],
                optimal_tool_calls=0, optimal_rows=0, optimal_vm_steps=0, split="live",
            )
            gen = {"max_new_tokens": 256, "do_sample": False, "temperature": None}
            ep = runner.run_episode(rollout_fn, tok, task, gen)
        return ep["calls"], ep["final_text"], ep["raw_completion"]


def default_backend() -> ModelBackend:
    return RunnerBackend()
