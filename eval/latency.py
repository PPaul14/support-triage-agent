"""Time classify-shaped phi3 calls: the compact prompt block first, the case last.

Only cache misses are timed, because a cached call measures a disk read, not inference.
Re-running therefore times nothing until those cache entries are removed.
"""

import json
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from src import classify, llm

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_JSONL = REPO_ROOT / "data" / "sample" / "cases_clean.jsonl"
N_CALLS = 10
SEED = 0
BUDGET_S = 15.0  # per call after the first: 200 cases x 4 systems has to fit in one night
PLANNED_CALLS = 200 * 4
INSTRUCTION = ("\nClassify the customer message below into exactly one intent from the list above. "
               'Reply with JSON only, in the form {"intent": "<intent name>"}.\n\n')


@dataclass
class Timing:
    case_id: int
    cache_hit: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    intent: str


def case_part(case: dict) -> str:
    """The per-case text. It goes LAST, so everything before it is a prefix Ollama can reuse."""
    lines = []
    for turn in case["prior_turns"][-2:]:  # the two most recent earlier turns, oldest first
        lines.append(f"Earlier {turn['role']} message: {turn['text']}")
    lines.append(f"Customer message: {case['customer_text']}")
    return "\n".join(lines)


def main() -> None:
    taxonomy = classify.TAXONOMY_MD.read_text(encoding="utf-8")
    block = classify.render_block(taxonomy)
    order, _ = classify.parse_taxonomy(taxonomy)
    schema = {"type": "object", "properties": {"intent": {"type": "string", "enum": order}}, "required": ["intent"]}
    quoted = set(int(case_id) for case_id in re.findall(r"\[case (\d+)\]", taxonomy))  # never the prompt's own examples
    cases = []
    with CLEAN_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            if case["case_id"] not in quoted:
                cases.append(case)
    timings = []
    for case in random.Random(SEED).sample(cases, N_CALLS):
        response = llm.complete(block + INSTRUCTION + case_part(case), classify.CLASSIFY_MODEL,
                                schema=schema, num_predict=32)
        timings.append(Timing(case_id=case["case_id"], cache_hit=response.cache_hit,
                              prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
                              latency_s=response.latency_s, intent=response.parsed["intent"]))

    print(f"{'call':>4} {'case_id':>8} {'cached':>6} {'prompt_tok':>10} {'output_tok':>10} {'latency_s':>9}  intent")
    for number, timing in enumerate(timings, start=1):
        print(f"{number:>4} {timing.case_id:>8} {str(timing.cache_hit):>6} {timing.prompt_tokens:>10} "
              f"{timing.completion_tokens:>10} {timing.latency_s:>9.1f}  {timing.intent}")
    misses = [timing for timing in timings if not timing.cache_hit]
    if len(misses) < 2:
        print("Fewer than 2 uncached calls, so nothing to time. Remove their llm_cache entries to re-measure.")
        return
    after_first = [timing.latency_s for timing in misses[1:]]
    median_s = statistics.median(after_first)
    print(f"first uncached call: {misses[0].latency_s:.1f} s (includes loading the model if it was not in memory)")
    print(f"after the first: median {median_s:.1f} s, max {max(after_first):.1f} s over {len(after_first)} calls")
    print(f"{PLANNED_CALLS} classify calls at that median: {PLANNED_CALLS * median_s / 3600:.1f} h")
    if median_s > BUDGET_S:
        print(f"ABOVE BUDGET: {median_s:.1f} s per call is more than {BUDGET_S:.0f} s")
    else:
        print(f"within budget: {median_s:.1f} s per call is under {BUDGET_S:.0f} s")


if __name__ == "__main__":
    main()
