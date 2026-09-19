"""The only module that talks to Ollama: disk cache, retries, schema repair and a call log."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "llm_cache"
CALL_LOG = CACHE_DIR.parent / "llm_calls.jsonl"
SEED = 0
MAX_ATTEMPTS = 3
TIMEOUT_S = (10, 600)  # (connect, read) - generation on CPU can take minutes
REPAIR_PROMPT = "Your reply could not be used: {reason}. Reply with only the corrected JSON."


@dataclass
class LLMResponse:
    text: str
    parsed: dict | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cache_hit: bool
    token_logprobs: list[dict] | None = None  # Ollama's {"token", "logprob", ...} per output token, if asked for


class LLMParseError(Exception):
    """A structured reply was still invalid after one repair attempt."""


def complete(prompt: str, model: str, *, system: str | None = None, schema: dict | None = None,
             temperature: float = 0.0, num_ctx: int = 4096, num_predict: int = 512,
             bypass_cache: bool = False, logprobs: bool = False) -> LLMResponse:
    """Run one LLM call, served from the disk cache when possible. Put the static prompt block FIRST
    and the per-case content LAST, so Ollama can reuse the already-processed prefix between calls.
    bypass_cache=True neither reads nor writes the cache: a temperature-0 re-run (a self-consistency
    check) must reach the model, and must not overwrite the reply that later stages read.
    logprobs=True also returns the log probability of every output token."""
    options = {"temperature": float(temperature), "num_ctx": num_ctx, "num_predict": num_predict, "seed": SEED}
    key = _cache_key(model, prompt, system, schema, options, logprobs)
    cache_path = CACHE_DIR / f"{key}.json"

    if cache_path.exists() and not bypass_cache:  # cache hit: return before any network code runs
        start = time.perf_counter()
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        response = LLMResponse(**stored["response"])
        response.cache_hit = True
        response.latency_s = time.perf_counter() - start
        _log_call(key, response, repair=False)
        return response

    messages = [{"role": "user", "content": prompt}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})

    response = _chat(model, messages, schema, options, logprobs)
    _log_call(key, response, repair=False)
    if schema is not None:
        try:
            response.parsed = _parse(response.text, schema)
        except ValueError as error:
            response = _repair(key, model, messages, schema, options, response, str(error), logprobs)

    if bypass_cache:
        return response
    # Only validated replies get here. Temp file + rename: a killed run never leaves half a file.
    request = {"model": model, "system": system, "prompt": prompt, "schema": schema, "options": options,
               "logprobs": logprobs}
    entry = {"request": request, "response": asdict(response)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(cache_path)
    return response


def _repair(key: str, model: str, messages: list[dict], schema: dict, options: dict,
            first: LLMResponse, reason: str, logprobs: bool) -> LLMResponse:
    """Send the invalid reply back once with the reason; raise if still invalid."""
    repair_messages = messages + [{"role": "assistant", "content": first.text},
                                  {"role": "user", "content": REPAIR_PROMPT.format(reason=reason)}]
    second = _chat(model, repair_messages, schema, options, logprobs)
    _log_call(key, second, repair=True)
    try:
        second.parsed = _parse(second.text, schema)
    except ValueError as error:
        raise LLMParseError(f"{model} returned unusable JSON twice. First: {reason}. After repair: "
                            f"{error}. Used {second.completion_tokens} of num_predict="
                            f"{options['num_predict']} tokens. Last reply: {second.text!r}") from error
    second.prompt_tokens += first.prompt_tokens  # the returned response carries the cost of both calls
    second.completion_tokens += first.completion_tokens
    second.latency_s += first.latency_s
    return second


def _chat(model: str, messages: list[dict], schema: dict | None, options: dict,
          logprobs: bool = False) -> LLMResponse:
    """POST one chat request to Ollama, retrying on connection errors and read timeouts."""
    payload = {"model": model, "messages": messages, "options": options, "stream": False,
               "logprobs": logprobs}
    if schema is not None:
        payload["format"] = schema
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(2 ** (attempt - 1))  # back off 1s, then 2s
        start = time.perf_counter()
        try:
            http_response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout) as error:
            last_error = error
            continue
        latency_s = time.perf_counter() - start
        if http_response.status_code >= 500:  # the model runner can crash mid-generation; that is retryable
            last_error = RuntimeError(f"Ollama returned HTTP {http_response.status_code}: {http_response.text}")
            continue
        if http_response.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {http_response.status_code}: {http_response.text}")
        body = http_response.json()
        return LLMResponse(text=body["message"]["content"], parsed=None, model=model,
                           prompt_tokens=body.get("prompt_eval_count", 0),
                           completion_tokens=body.get("eval_count", 0),
                           latency_s=latency_s, cache_hit=False, token_logprobs=body.get("logprobs"))
    raise ConnectionError(f"Could not reach Ollama at {OLLAMA_URL} after {MAX_ATTEMPTS} attempts") from last_error


def _parse(text: str, schema: dict) -> dict:
    """Parse and validate a structured reply; raise ValueError with a short reason."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"not valid JSON ({error})") from error
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as error:
        raise ValueError(f"does not match the schema ({error.message})") from error
    return parsed


def _cache_key(model: str, prompt: str, system: str | None, schema: dict | None, options: dict,
               logprobs: bool = False) -> str:
    """sha256 over everything that changes the output. json.dumps, not "|".join: no collisions.
    logprobs joins the key only when asked for, so every key written before it existed still matches."""
    parts = [model, prompt, system, schema, options]
    if logprobs:
        parts.append("logprobs")
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _log_call(key: str, response: LLMResponse, repair: bool) -> None:
    """Append one line per call to the log the cost report reads (reply text left out)."""
    # Cost and throughput reports must keep cache_hit=False rows only: a cached row's latency_s
    # is a disk read, not inference, and its token counts repeat the original call's.
    record = {"time": datetime.now(timezone.utc).isoformat(), "key": key, "repair": repair,
              "model": response.model, "cache_hit": response.cache_hit, "prompt_tokens": response.prompt_tokens,
              "completion_tokens": response.completion_tokens, "latency_s": round(response.latency_s, 3)}
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CALL_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")
