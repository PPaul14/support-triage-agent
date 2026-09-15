"""Golden set, part 2: the labelling CLI (python -m src.label [--assist]). Resumable: each answer is saved at once.

--assist shows phi3's proposed intent to accept or override, except at every 5th pool position, which stays blind."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import classify, llm

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_JSONL = REPO_ROOT / "data" / "sample" / "cases_clean.jsonl"
POOL_JSONL = REPO_ROOT / "data" / "golden" / "golden_pool.jsonl"
GOLDEN_JSONL = REPO_ROOT / "data" / "golden" / "golden_set.jsonl"
BLIND_EVERY = 5  # with --assist, pool positions 5, 10, 15, ... are labelled with the suggestion hidden
KEY_BLOCK = ("  1 billing  2 login  3 playback  4 playlists/downloads  5 not available\n"
             "  6 feature request  7 reply, problem not named  8 thanks/chatter  9 unclear\n"
             "  compromised: n unless someone else got into their account\n"
             "  escalate: y for 1, 2 with hacking, 9, and 7 without enough context. n otherwise.\n"
             "  difficulty: 1 obvious  2 paused  3 torn        notes: Enter to skip")
REASONS = {"1": "needs account data", "2": "account compromised", "3": "cannot determine intent"}


@dataclass
class Label:
    case_id: int
    intent: str
    compromised: bool  # someone else got into the account: the SEVERE harm rule reads it as data
    escalate: bool
    escalate_reason: str  # empty when escalate is False
    difficulty: int  # 1 easy, 2 medium, 3 hard
    notes: str
    labelled_at: str  # ISO 8601, UTC
    proposed_intent: str | None = None  # phi3's suggestion, recorded even when hidden; None without --assist
    accepted: bool | None = None  # my intent equals the shown suggestion; None when no suggestion was shown
    mode: str = "manual"  # "assisted" (suggestion shown), "blind" (suggestion hidden) or "manual" (no --assist)


def ask(prompt: str, allowed: list[str] | None = None) -> str:
    """Read one answer, repeating until it is in `allowed` when given. 'q' quits; everything so far is saved."""
    while True:
        answer = input(prompt).strip()
        if answer.lower() == "q":
            raise SystemExit("Stopped. Run python -m src.label again to resume where you left off.")
        if allowed is None or answer.lower() in allowed:
            return answer
        print(f"  Please answer one of: {', '.join(allowed)}")


def show(case: dict, position: int, total: int, intents: list[str], counts: Counter[str]) -> None:
    """Print the case the way the labeller sees it: prior turns and the customer message, nothing else."""
    print(f"\n{'=' * 72}\ncase {position}/{total}   (case_id {case['case_id']})")
    print(KEY_BLOCK)
    if case["prior_turns"]:
        print("\nEarlier in the thread:")
        for turn in case["prior_turns"]:
            print(f"  [{turn['role']}] {turn['text']}")
    print(f"\nCustomer message:\n  {case['customer_text']}\n")
    for number, intent in enumerate(intents, start=1):
        print(f"  {number}  {intent:22} ({counts[intent]} so far)")


def suggestions(cases: list[dict], intents: list[str]) -> dict[int, str]:
    """phi3's proposed intent per case: the sampler's cached estimate where there is one, otherwise a live call."""
    block = classify.render_block(classify.TAXONOMY_MD.read_text(encoding="utf-8"))
    schema = classify.intent_schema(intents)
    proposed: dict[int, str] = {}
    from_cache = 0
    for case in cases:  # all of them back to back under one model, before any labelling starts
        try:
            response = llm.complete(classify.intent_prompt(case, block), classify.CLASSIFY_MODEL,
                                    schema=schema, num_predict=32)
        except llm.LLMParseError:  # no usable suggestion: this case is labelled as if without --assist
            continue
        proposed[case["case_id"]] = response.parsed["intent"]
        if response.cache_hit:
            from_cache += 1
    print(f"Suggestions ready for {len(proposed)} cases: {from_cache} from the cache, "
          f"{len(proposed) - from_cache} new phi3 calls.")
    return proposed


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-label the golden pool, one case at a time.")
    parser.add_argument("--assist", action="store_true", help="suggest phi3's intent; every 5th case stays blind")
    assist = parser.parse_args().assist
    sys.stdout.reconfigure(encoding="utf-8")  # tweets contain emoji; a Windows pipe defaults to cp1252
    intents, _ = classify.parse_taxonomy(classify.TAXONOMY_MD.read_text(encoding="utf-8"))
    pool_ids = [json.loads(line)["case_id"] for line in POOL_JSONL.read_text(encoding="utf-8").splitlines()]
    wanted = set(pool_ids)
    all_cases = [json.loads(line) for line in CLEAN_JSONL.read_text(encoding="utf-8").splitlines()]
    cases = {case["case_id"]: case for case in all_cases if case["case_id"] in wanted}
    done: set[int] = set()
    counts: Counter[str] = Counter()
    assisted_labels: list[Label] = []  # for the running override rate
    if GOLDEN_JSONL.exists():
        with GOLDEN_JSONL.open(encoding="utf-8") as handle:
            for line in handle:
                label = Label(**json.loads(line))  # records written before --assist existed load with the defaults
                done.add(label.case_id)
                counts[label.intent] += 1
                if label.mode == "assisted":
                    assisted_labels.append(label)
    print(f"{len(done)}/{len(pool_ids)} already labelled. Type q at any prompt to stop; nothing is lost.\n{KEY_BLOCK}")
    proposed = suggestions([cases[i] for i in pool_ids if i not in done], intents) if assist else {}

    for position, case_id in enumerate(pool_ids, start=1):  # the pool's order is the labelling order
        if case_id in done:
            continue
        blind = position % BLIND_EVERY == 0
        suggestion = None if blind else proposed.get(case_id)
        show(cases[case_id], len(done) + 1, len(pool_ids), intents, counts)
        choices = [str(n) for n in range(1, len(intents) + 1)]
        if suggestion is not None:
            print(f"\n  suggested: {suggestion}  [Enter to accept]")
            choices.append("")  # Enter accepts the suggestion; a digit overrides it
        number = ask(f"intent [1-{len(intents)}]: ", choices)
        intent = suggestion if number == "" else intents[int(number) - 1]
        compromised = ask("compromised account, someone else got in [y/n]: ", ["y", "n"]).lower() == "y"
        escalate = ask("escalate [y/n]: ", ["y", "n"]).lower() == "y"
        reason = ask("escalate reason [1 needs account data, 2 account compromised, 3 cannot determine intent, "
                     "or type it]: ") if escalate else ""
        reason = REASONS.get(reason, reason)  # a single digit 1-3 is a shortcut; anything else is kept as typed
        difficulty = ask("difficulty [1 easy, 2 medium, 3 hard]: ", ["1", "2", "3"])
        notes = ask("notes (Enter to skip): ")
        mode = "blind" if blind else "assisted"
        if case_id not in proposed:  # no --assist, or no usable suggestion for this case
            mode = "manual"
        label = Label(case_id=case_id, intent=intent, compromised=compromised, escalate=escalate,
                      escalate_reason=reason, difficulty=int(difficulty), notes=notes,
                      labelled_at=datetime.now(timezone.utc).isoformat(), proposed_intent=proposed.get(case_id),
                      accepted=None if suggestion is None else intent == suggestion, mode=mode)
        with GOLDEN_JSONL.open("a", encoding="utf-8", newline="\n") as handle:  # saved before the next case
            handle.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
        done.add(case_id)
        counts[label.intent] += 1
        if mode == "assisted":
            assisted_labels.append(label)
            overridden = sum(1 for assisted in assisted_labels if not assisted.accepted)
            print(f"  override rate so far: {overridden} of {len(assisted_labels)} assisted cases")
    print(f"\nAll {len(pool_ids)} cases labelled: {dict(counts.most_common())}")


if __name__ == "__main__":
    main()
