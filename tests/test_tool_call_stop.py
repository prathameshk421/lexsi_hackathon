"""The runner stops a model turn at the end of a complete tool call.

Part 1 tests the pure check `ends_with_tool_call`. Part 2 drives AgentTune's
real rollout loop and our engine with a tiny random model whose `generate` is
scripted, so no weights are downloaded.
"""

from __future__ import annotations

import pytest

from frugal_analyst.reward import ToolCall
from frugal_analyst.trajectory import ends_with_tool_call, format_tool_call_turn, parse_completion

CALL = '[{"tool_name": "get_schema", "parameters": {"database": "clinic"}}]'
SQL_CALL = format_tool_call_turn([ToolCall("run_sql", {"database": "retail", "query": "SELECT COUNT(*) FROM orders WHERE note = '}]' OR tag = '[x]'"})])


# ---------------------------------------------------------------------------
# Part 1: ends_with_tool_call
# ---------------------------------------------------------------------------


STOP_TEXTS = [
    CALL,
    CALL + "\n",
    "  " + CALL + " \n\n",
    "I will look at the schema first.\n" + CALL,
    SQL_CALL,
    '[{"tool_name": "get_schema", "parameters": {"database": "clinic"}}, '
    '{"tool_name": "run_sql", "parameters": {"database": "clinic", "query": "SELECT 1"}}]',
    "```json\n" + CALL + "\n```",
    "```json\n" + CALL,  # the stop fires before the fence closes
    '[{"tool_name": "run_sql", "parameters": {"database": "x", "query": "SELECT json_extract(a, \'$.b[0]\') FROM t"}}]',
    '[{"name": "get_schema", "arguments": {"database": "clinic"}}]',  # generic shape, also parsed
]


@pytest.mark.parametrize("text", STOP_TEXTS)
def test_complete_call_at_the_end(text):
    assert ends_with_tool_call(text)


@pytest.mark.parametrize("text", STOP_TEXTS)
def test_parser_reads_every_text_that_stops(text):
    # The stop rule must never fire on text the parser does not read as a call.
    pytest.importorskip("agenttune")
    calls, _ = parse_completion(text)
    assert calls


@pytest.mark.parametrize("text", [
    "",
    None,
    CALL[:-1],  # list not closed yet
    CALL[:-3],
    SQL_CALL[: SQL_CALL.index("}]'") + 2],  # partial: ends with "}]" inside the SQL string
    SQL_CALL[: SQL_CALL.index("[x]") + 3],  # partial: ends with "[x]" inside the SQL string
    CALL + "\nTool result: 1\nANSWER: 1",  # text after the list
    CALL + "ANSWER: 1",
    "ANSWER: 42",
    "ANSWER: [1, 2, 3]",  # a list, but not of calls
    "[]",
    '[{"database": "clinic"}]',  # dicts without a tool name
    '{"tool_name": "get_schema", "parameters": {"database": "clinic"}}',  # not a list
    "```json\n" + CALL + "\n```\nDone.",
])
def test_no_complete_call_at_the_end(text):
    assert not ends_with_tool_call(text)


def test_check_is_false_until_the_list_closes():
    # Feed a call one character at a time, as a model writes it.
    hits = [i for i in range(1, len(SQL_CALL) + 1) if ends_with_tool_call(SQL_CALL[:i])]
    assert hits == [len(SQL_CALL)]


# ---------------------------------------------------------------------------
# Part 2: the engine with a scripted tiny model
# ---------------------------------------------------------------------------

FAKE = "\nTool result: 12345\nANSWER: 12345"


@pytest.fixture(scope="module")
def tok():
    pytest.importorskip("agenttune")
    from frugal_analyst.runner import load_tokenizer

    try:
        from huggingface_hub import try_to_load_from_cache

        if not isinstance(try_to_load_from_cache("HuggingFaceTB/SmolLM2-135M-Instruct", "tokenizer_config.json"), str):
            pytest.skip("SmolLM2 tokenizer is not in the local HF cache")
        return load_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct")
    except OSError:
        pytest.skip("SmolLM2 tokenizer is not in the local HF cache")


def scripted_model(tok, turns: list[str]):
    """A 1-layer random Llama whose generate() writes `turns`, one per call.

    It appends one token at a time and checks eos and the stopping criteria
    after each token, as transformers' generate() does.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    class ScriptedLM(LlamaForCausalLM):
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=64, eos_token_id=None,
                     stopping_criteria=None, **kwargs):
            self.prompts.append(input_ids[0].tolist())
            turn = len(self.prompts) - 1
            text = turns[turn] if turn < len(turns) else ""
            script = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
            eos = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
            ids = input_ids
            for t in script[:max_new_tokens]:
                ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
                if t in eos:
                    break
                if stopping_criteria is not None and bool(stopping_criteria(ids, None).all()):
                    break
            return ids

    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=8192)
    model = ScriptedLM(cfg)
    model.prompts = []
    model.eval()
    return model


def run_one(tok, stop: bool):
    from frugal_analyst.db import ensure_dbs
    from frugal_analyst.runner import build_rollout_fn, run_episode
    from frugal_analyst.trajectory import DEFAULT_TASKS_DIR
    from frugal_analyst.tasks import load_tasks

    ensure_dbs()
    task = next(t for t in load_tasks(DEFAULT_TASKS_DIR / "test_id.jsonl") if t.domain == "clinic")
    model = scripted_model(tok, [CALL + FAKE, "ANSWER: 7"])
    rollout_fn = build_rollout_fn(model, tok, max_tool_steps=4, stop_after_tool_call=stop)
    ep = run_episode(rollout_fn, tok, task, {"max_new_tokens": 128, "do_sample": False, "temperature": None})
    return model, rollout_fn, ep


def test_stop_ends_the_turn_at_the_call(tok):
    model, rollout_fn, ep = run_one(tok, stop=True)
    conv = ep["trajectory"].metadata["conversation"]
    assistant = [m["content"] for m in conv if m.get("role") == "assistant"]
    assert assistant[0] == CALL  # the kept turn ends at the tool call
    assert assistant[-1] == "ANSWER: 7"
    second_prompt = tok.decode(model.prompts[1])
    assert "Tool result for get_schema" in second_prompt  # the real tool result is there
    assert "12345" not in second_prompt  # the fake text is not
    assert [c.name for c in ep["calls"]] == ["get_schema"]
    assert ep["final_text"] == "ANSWER: 7"
    assert "12345" not in ep["raw_completion"]
    assert rollout_fn.engine.stats == {"turns": 2, "stopped_after_tool_call": 1}
    assert "generate" not in model.__dict__  # the per-turn wrapper is removed


def test_no_stop_keeps_the_old_behaviour(tok):
    model, rollout_fn, ep = run_one(tok, stop=False)
    second_prompt = tok.decode(model.prompts[1])
    assert "12345" in second_prompt  # the fake text stays in the context
    assert "12345" in ep["raw_completion"]
    assert rollout_fn.engine.stats["stopped_after_tool_call"] == 0


def test_run_log_path_is_not_a_trace_file(tmp_path):
    from frugal_analyst.runner import run_log_path

    assert run_log_path(tmp_path / "zero-shot_test_id.jsonl") == tmp_path / "zero-shot_test_id.run.json"
    assert not run_log_path("a/b.jsonl").name.endswith(".jsonl")
