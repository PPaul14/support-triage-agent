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
