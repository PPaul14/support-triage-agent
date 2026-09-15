"""Golden set, part 1: sample the 150 cases to hand-label (python -m src.golden). Makes no model call."""

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from src import classify, llm

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_JSONL = REPO_ROOT / "data" / "sample" / "cases_clean.jsonl"
POOL_JSONL = REPO_ROOT / "data" / "golden" / "golden_pool.jsonl"
SEED = 0
GOLDEN_SIZE = 150
ESTIMATE_POOL_SIZE = 1500
HARD_PER_RULE = 6
MIN_PER_INTENT = 20  # cached estimates an intent needs before its ~13 stratified cases are drawn
HARD_RULES = ["non_english", "screenshot_only", "charge_and_failure", "short_reply", "longest"]
PLACEHOLDERS = {"@USER", "URL"}
CHARGE = re.compile(r"\b(charg(e|ed|es|ing)|paid|payments?|billed|billing|refund(ed|s)?|debit(ed)?)\b", re.IGNORECASE)
FAILURE = re.compile(r"\b(bugs?|crash(es|ed|ing)?|errors?|broken|glitch\w*|freez\w*|frozen|not working|stopped "
                     r"(working|playing)|(doesn'?t|won'?t|isn'?t|can'?t) (work|play|load|open)(ing)?)\b", re.IGNORECASE)
INSTRUCTION = ("\nEstimate which intent from the list above fits the customer message below. "
               'Reply with JSON only, in the form {"intent": "<intent name>"}.\n\n')


@dataclass
class PoolEntry:
    case_id: int
    thread_id: int
    dup_group_id: int | None
    stratum: str  # "hard:<rule>" or "estimated:<intent>"; the estimate is phi3's, never a label


def load_eligible(taxonomy: str) -> tuple[list[dict], set[int], set[int], set[int]]:
    """Cases that may enter the golden set, and the excluded example case_ids, thread_ids and dup groups."""
    cases = [json.loads(line) for line in CLEAN_JSONL.read_text(encoding="utf-8").splitlines()]
    example_ids = set(int(case_id) for case_id in re.findall(r"\[case (\d+)\]", taxonomy))
    threads = set(case["thread_id"] for case in cases if case["case_id"] in example_ids)
    groups = set(case["dup_group_id"] for case in cases if case["case_id"] in example_ids) - {None}
    dropped: Counter[str] = Counter()
    eligible = []
    for case in cases:
        if case["case_id"] in example_ids:
            dropped["taxonomy example"] += 1
        elif case["thread_id"] in threads:
            dropped["same thread as an example"] += 1
        elif case["dup_group_id"] in groups:
            dropped["same duplicate group as an example"] += 1
        elif case["dup_group_id"] not in (None, case["case_id"]):
            dropped["near-duplicate of a kept case"] += 1  # one per group: the group's smallest case_id stays
        else:
            eligible.append(case)
    print(f"eligible {len(eligible):,} of {len(cases):,} | excluded {dict(dropped)} | exclusion sets: "
          f"{len(example_ids)} case_ids, {len(threads)} thread_ids, {len(groups)} duplicate group(s)")
    return eligible, example_ids, threads, groups


def matches(rule: str, case: dict) -> bool:
    """Whether a case qualifies for a hard-case rule. 'longest' ranks every case, so every case qualifies."""
    text = case["customer_text"]
    content_words = sum(1 for token in text.split() if token not in PLACEHOLDERS)
    checks = {
        "non_english": case["lang"] not in ("en", "unknown"),  # a real code; "unknown" is mostly short English
        "screenshot_only": re.search(r"\bURL\b", text) is not None and content_words < 6,
        "charge_and_failure": CHARGE.search(text) is not None and FAILURE.search(text) is not None,
        "short_reply": len(case["prior_turns"]) > 0 and content_words < 5,
    }
    return checks.get(rule, True)


def pick_hard(eligible: list[dict], rng: random.Random) -> dict[int, str]:
    """case_id -> stratum: HARD_PER_RULE cases per rule, rules in order, no case picked twice."""
    picked: dict[int, str] = {}
    for rule in HARD_RULES:
        candidates = [case for case in eligible if case["case_id"] not in picked and matches(rule, case)]
        if rule == "longest":
            candidates.sort(key=lambda case: (-len(case["customer_text"]), case["case_id"]))
            chosen = candidates[:HARD_PER_RULE]
        else:
            chosen = rng.sample(candidates, min(HARD_PER_RULE, len(candidates)))
        print(f"hard rule {rule}: {len(candidates):,} candidates, {len(chosen)} picked")
        for case in chosen:
            picked[case["case_id"]] = f"hard:{rule}"
    return picked


def cached_estimates(cases: list[dict], block: str, order: list[str]) -> dict[int, str]:
    """phi3's intent ESTIMATE per case, read from the disk cache: no model call; uncached cases are skipped."""
    schema = {"type": "object", "properties": {"intent": {"type": "string", "enum": order}}, "required": ["intent"]}
    options = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 32, "seed": llm.SEED}  # exactly what complete() sent
    estimates: dict[int, str] = {}
    for case in cases:
        lines = [f"Earlier {turn['role']} message: {turn['text']}" for turn in case["prior_turns"][-2:]]
        lines.append(f"Customer message: {case['customer_text']}")
        key = llm._cache_key(classify.CLASSIFY_MODEL, block + INSTRUCTION + "\n".join(lines), None, schema, options)
        cache_path = llm.CACHE_DIR / f"{key}.json"
        if cache_path.exists():  # a drifted prompt or option reads as "not cached", never as a model call
            stored = json.loads(cache_path.read_text(encoding="utf-8"))
            estimates[case["case_id"]] = stored["response"]["parsed"]["intent"]
    return estimates


def pick_stratified(estimates: dict[int, str], order: list[str], taken: set[int], size: int,
                    rng: random.Random) -> dict[int, str]:
    """case_id -> stratum: round-robin over intents in taxonomy order, so each gets about size / 9 cases."""
    candidates = sorted(case_id for case_id in estimates if case_id not in taken)
    rng.shuffle(candidates)
    queues = {intent: [case_id for case_id in candidates if estimates[case_id] == intent] for intent in order}
    short = {intent: MIN_PER_INTENT - len(queue) for intent, queue in queues.items() if len(queue) < MIN_PER_INTENT}
    assert not short, f"fewer than {MIN_PER_INTENT} cached estimates for (intent: cases missing): {short}"
    picked: dict[int, str] = {}
    while len(picked) < size and any(queues.values()):
        for intent in order:
            if queues[intent] and len(picked) < size:
                picked[queues[intent].pop()] = f"estimated:{intent}"
    return picked


def main() -> None:
    taxonomy = classify.TAXONOMY_MD.read_text(encoding="utf-8")
    order, _ = classify.parse_taxonomy(taxonomy)
    eligible, example_ids, threads, groups = load_eligible(taxonomy)
    rng = random.Random(SEED)
    estimate_pool = rng.sample(eligible, ESTIMATE_POOL_SIZE)  # drawn first, so a hard-rule change cannot alter it
    strata = pick_hard(eligible, rng)
    estimates = cached_estimates(estimate_pool, classify.render_block(taxonomy), order)
    print(f"cached phi3 estimates, NOT labels: {len(estimates):,} of {ESTIMATE_POOL_SIZE:,} | "
          f"{dict(Counter(estimates.values()).most_common())}")
    strata.update(pick_stratified(estimates, order, set(strata), GOLDEN_SIZE - len(strata), rng))
    by_id = {case["case_id"]: case for case in eligible}
    entries = [PoolEntry(case_id, by_id[case_id]["thread_id"], by_id[case_id]["dup_group_id"], strata[case_id])
               for case_id in sorted(strata)]
    rng.shuffle(entries)  # labelling order: strata interleaved, so the hard cases do not arrive as one block
    assert len(entries) == GOLDEN_SIZE, f"sampled {len(entries)} cases, expected {GOLDEN_SIZE}"
    for entry in entries:  # the exclusions must hold for every sampled case
        assert entry.case_id not in example_ids and entry.thread_id not in threads and entry.dup_group_id not in groups
    dup_groups = [entry.dup_group_id for entry in entries if entry.dup_group_id is not None]
    assert len(dup_groups) == len(set(dup_groups)), "two sampled cases share a duplicate group"
    with POOL_JSONL.open("w", encoding="utf-8", newline="\n") as handle:  # data/golden/ is part of the repo
        for entry in entries:
            handle.write(json.dumps(asdict(entry)) + "\n")
    print(f"exclusion assertions passed | wrote {POOL_JSONL}: {dict(sorted(Counter(e.stratum for e in entries).items()))}")


if __name__ == "__main__":
    main()
