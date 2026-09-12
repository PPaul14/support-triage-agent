"""Time each stage's LLM calls: classify (phi3), draft (llama3) and judge (qwen2.5 and mistral).

Each model's calls run as one back-to-back batch, never interleaved, with the static prompt part first
and the per-case part last so Ollama can reuse the prefix. Only cache misses are timed: a cached call
measures a disk read, not inference. The draft and judge prompts are latency-shaped placeholders
(random precedents, a stand-in rubric), not the prompts the evaluation will use.
"""

import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from src import classify, llm

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_JSONL = REPO_ROOT / "data" / "sample" / "cases_clean.jsonl"
N_CALLS = 10
N_PRECEDENTS = 3
SEED = 0
DRAFT_MODEL = "llama3:8b-instruct-q4_0"
JUDGE_MODELS = ["qwen2.5:7b-instruct-q4_K_M", "mistral:7b-instruct-v0.3-q4_K_M"]
# Calls in a full overnight run: 200 cases x 4 systems classified and drafted, 200 x 3 systems per judge.
PLANNED_CALLS = {"classify": 200 * 4, "draft": 200 * 4, "judge": 200 * 3}
CLASSIFY_INSTRUCTION = ("\nClassify the customer message below into exactly one intent from the list above. "
                        'Reply with JSON only, in the form {"intent": "<intent name>"}.\n\n')
DRAFT_INSTRUCTION = ("\nWrite the brand's reply to the customer message at the end, in under 280 characters. "
                     "Use the past cases as precedents for facts and tone.\n\n")
CHECKS = ["addresses_issue", "grounded", "no_false_promise", "handling_policy", "tone"]
RUBRIC = ("Grade the drafted support reply at the end. For each check answer yes or no, and quote the words "
          "(under 10) from the draft or the case that decide it.\n"
          "addresses_issue: does the draft answer the problem the customer actually raised?\n"
          "grounded: is every factual claim in the draft supported by the case or the past cases?\n"
          "no_false_promise: does the draft avoid promising refunds, account changes or fixes it cannot make?\n"
          "handling_policy: if the case needs a human, does the draft say it is being passed on?\n"
          "tone: is the draft polite, brief and in the brand's voice?\n\n")


@dataclass
class Timing:
    case_id: int
    cache_hit: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    failed: bool  # the reply was unusable even after one repair; its time still counts


def case_part(case: dict) -> str:
    """The per-case text. It goes LAST, so everything before it is a prefix Ollama can reuse."""
    lines = []
    for turn in case["prior_turns"][-2:]:  # the two most recent earlier turns, oldest first
        lines.append(f"Earlier {turn['role']} message: {turn['text']}")
    lines.append(f"Customer message: {case['customer_text']}")
    return "\n".join(lines)


def pick_precedents(case: dict, pool: list[dict], rng: random.Random) -> list[dict]:
    """Stand-ins for retrieval: random other cases, never from the same thread or near-duplicate group."""
    chosen: list[dict] = []
    while len(chosen) < N_PRECEDENTS:
        candidate = rng.choice(pool)
        same_thread = candidate["thread_id"] == case["thread_id"]
        same_group = case["dup_group_id"] is not None and candidate["dup_group_id"] == case["dup_group_id"]
        if not same_thread and not same_group and candidate not in chosen:
            chosen.append(candidate)
    return chosen


def precedent_part(precedents: list[dict]) -> str:
    """Past cases with the brand's real reply. They differ per case, so they come after the static part."""
    lines = ["Past cases:"]
    for precedent in precedents:
        lines.append(f"Customer: {precedent['customer_text']}")
        lines.append(f"Brand reply: {precedent['brand_reply']}")
    return "\n".join(lines)


def judge_schema() -> dict:
    check = {"type": "object", "required": ["answer", "evidence"],
             "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}, "evidence": {"type": "string"}}}
    return {"type": "object", "properties": {name: check for name in CHECKS}, "required": CHECKS}


def run_batch(model: str, prompts: list[tuple[int, str]], schema: dict | None,
              num_predict: int) -> tuple[list[Timing], list[str]]:
    """One model's calls, back to back. Returns the timings and the reply texts ("" when unusable)."""
    timings: list[Timing] = []
    texts: list[str] = []
    for case_id, prompt in prompts:
        start = time.perf_counter()
        try:
            response = llm.complete(prompt, model, schema=schema, num_predict=num_predict)
        except llm.LLMParseError:
            timings.append(Timing(case_id, False, 0, 0, time.perf_counter() - start, failed=True))
            texts.append("")
            continue
        timings.append(Timing(case_id, response.cache_hit, response.prompt_tokens, response.completion_tokens,
                              response.latency_s, failed=False))
        texts.append(response.text)
    return timings, texts


def report(label: str, timings: list[Timing], planned_calls: int) -> float | None:
    """Print one batch. Returns the hours planned_calls would take, or None if too few calls were uncached."""
    print(f"\n== {label}")
    print(f"{'call':>4} {'case_id':>8} {'cached':>6} {'prompt_tok':>10} {'output_tok':>10} {'latency_s':>9} failed")
    for number, t in enumerate(timings, start=1):
        print(f"{number:>4} {t.case_id:>8} {str(t.cache_hit):>6} {t.prompt_tokens:>10} {t.completion_tokens:>10} "
              f"{t.latency_s:>9.1f} {t.failed}")
    misses = [t for t in timings if not t.cache_hit]
    if len(misses) < 2:
        print("fewer than 2 uncached calls, so nothing to time")
        return None
    after_first = [t.latency_s for t in misses[1:]]
    median_s = statistics.median(after_first)
    hours = (misses[0].latency_s + (planned_calls - 1) * median_s) / 3600  # the first call carries the model load
    print(f"first {misses[0].latency_s:.1f} s | after the first: median {median_s:.1f} s, max {max(after_first):.1f} s"
          f" | unusable {sum(1 for t in timings if t.failed)} | {planned_calls} calls: {hours:.1f} h")
    return hours


def main() -> None:
    taxonomy = classify.TAXONOMY_MD.read_text(encoding="utf-8")
    block = classify.render_block(taxonomy)
    order, _ = classify.parse_taxonomy(taxonomy)
    quoted = set(int(case_id) for case_id in re.findall(r"\[case (\d+)\]", taxonomy))  # never the prompt's own examples
    cases = []
    with CLEAN_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            if case["case_id"] not in quoted:
                cases.append(case)
    rng = random.Random(SEED)
    sample = rng.sample(cases, N_CALLS)  # the first draw, so the classify prompts match earlier runs
    precedents = {case["case_id"]: pick_precedents(case, cases, rng) for case in sample}

    hours = {}
    classify_schema = {"type": "object", "properties": {"intent": {"type": "string", "enum": order}},
                       "required": ["intent"]}
    prompts = [(c["case_id"], block + CLASSIFY_INSTRUCTION + case_part(c)) for c in sample]
    timings, _ = run_batch(classify.CLASSIFY_MODEL, prompts, classify_schema, 32)
    hours["classify"] = report(f"classify: {classify.CLASSIFY_MODEL}", timings, PLANNED_CALLS["classify"])

    prompts = [(c["case_id"], block + DRAFT_INSTRUCTION + precedent_part(precedents[c["case_id"]]) + "\n\n"
                + case_part(c) + "\nBrand reply:") for c in sample]
    timings, drafts = run_batch(DRAFT_MODEL, prompts, None, 120)
    hours["draft"] = report(f"draft: {DRAFT_MODEL}", timings, PLANNED_CALLS["draft"])

    prompts = [(c["case_id"], RUBRIC + precedent_part(precedents[c["case_id"]]) + "\n\n" + case_part(c)
                + f"\nDrafted reply: {draft}") for c, draft in zip(sample, drafts)]
    for model in JUDGE_MODELS:  # one judge's whole batch, then the other's: never interleaved
        timings, _ = run_batch(model, prompts, judge_schema(), 200)
        hours[model] = report(f"judge: {model}", timings, PLANNED_CALLS["judge"])

    print()
    for stage, stage_hours in hours.items():
        print(f"{stage:34} {'not timed (cached)' if stage_hours is None else f'{stage_hours:.1f} h'}")
    print(f"{'total of the stages timed here':34} {sum(h for h in hours.values() if h is not None):.1f} h")


if __name__ == "__main__":
    main()
