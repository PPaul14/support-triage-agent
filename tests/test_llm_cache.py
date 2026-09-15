"""The disk cache: a repeated identical call must be served from disk, not from Ollama."""

import json
from pathlib import Path

import pytest
import requests

from src import llm

MODEL = "phi3:3.8b-mini-128k-instruct-q4_0"
PROMPT = "In one short sentence, what does a customer support agent do?"


def _ollama_reachable() -> bool:
    """True if the Ollama server answers at all. Whether MODEL is pulled is not checked."""
    server_root = llm.OLLAMA_URL.removesuffix("/api/chat")
    try:
        requests.get(server_root, timeout=2)
    except requests.RequestException:
        return False
    return True


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama server not reachable")
def test_second_identical_call_is_a_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fresh cache and log per run: the first call is always a real miss, and the
    # project's own artifacts/ folder is never touched by the test.
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "CALL_LOG", tmp_path / "llm_calls.jsonl")

    first = llm.complete(PROMPT, MODEL)
    second = llm.complete(PROMPT, MODEL)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == first.text
    assert second.prompt_tokens == first.prompt_tokens
    assert second.completion_tokens == first.completion_tokens

    log_lines = (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in log_lines]
    assert [record["cache_hit"] for record in records] == [False, True]
    assert records[0]["key"] == records[1]["key"]


def test_bypass_cache_reaches_the_model_and_leaves_the_cache_alone(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake model call instead of Ollama, so this runs in CI: each call returns a new reply text.
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "CALL_LOG", tmp_path / "llm_calls.jsonl")
    calls = []

    def fake_chat(model: str, messages: list[dict], schema: dict | None, options: dict,
                  logprobs: bool = False) -> llm.LLMResponse:
        calls.append(model)
        return llm.LLMResponse(text=f"reply {len(calls)}", parsed=None, model=model, prompt_tokens=1,
                               completion_tokens=1, latency_s=0.0, cache_hit=False)

    monkeypatch.setattr(llm, "_chat", fake_chat)
    first = llm.complete(PROMPT, MODEL)
    cached = llm.complete(PROMPT, MODEL)
    rerun = llm.complete(PROMPT, MODEL, bypass_cache=True)
    after = llm.complete(PROMPT, MODEL)

    assert len(calls) == 2  # only the first call and the bypassing re-run reached the model
    assert first.text == "reply 1" and cached.cache_hit and cached.text == "reply 1"
    assert not rerun.cache_hit and rerun.text == "reply 2"
    assert after.cache_hit and after.text == "reply 1"  # the re-run did not overwrite the cached reply
