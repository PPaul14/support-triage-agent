"""Golden set, part 2: the labelling CLI (python -m src.label). Resumable: each answer is appended at once.

It shows the customer message and its prior turns only. The brand's real reply, the language tag and the
sampler's intent estimate are never shown, because each of them would anchor the label.
"""

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import classify

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_JSONL = REPO_ROOT / "data" / "sample" / "cases_clean.jsonl"
POOL_JSONL = REPO_ROOT / "data" / "golden" / "golden_pool.jsonl"
GOLDEN_JSONL = REPO_ROOT / "data" / "golden" / "golden_set.jsonl"


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
    if case["prior_turns"]:
        print("\nEarlier in the thread:")
        for turn in case["prior_turns"]:
            print(f"  [{turn['role']}] {turn['text']}")
    print(f"\nCustomer message:\n  {case['customer_text']}\n")
    for number, intent in enumerate(intents, start=1):
        print(f"  {number}  {intent:22} ({counts[intent]} so far)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # tweets contain emoji; a Windows pipe defaults to cp1252
    intents, _ = classify.parse_taxonomy(classify.TAXONOMY_MD.read_text(encoding="utf-8"))
    pool_ids = []
    with POOL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            pool_ids.append(json.loads(line)["case_id"])
    wanted = set(pool_ids)
    cases = {}
    with CLEAN_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            if case["case_id"] in wanted:
                cases[case["case_id"]] = case
    done: set[int] = set()
    counts: Counter[str] = Counter()
    if GOLDEN_JSONL.exists():
        with GOLDEN_JSONL.open(encoding="utf-8") as handle:
            for line in handle:
                label = Label(**json.loads(line))
                done.add(label.case_id)
                counts[label.intent] += 1
    print(f"{len(done)}/{len(pool_ids)} already labelled. Type q at any prompt to stop; nothing is lost.")

    for case_id in pool_ids:  # the pool's order is the labelling order
        if case_id in done:
            continue
        show(cases[case_id], len(done) + 1, len(pool_ids), intents, counts)
        number = ask(f"intent [1-{len(intents)}]: ", [str(n) for n in range(1, len(intents) + 1)])
        compromised = ask("compromised account, someone else got in [y/n]: ", ["y", "n"]).lower() == "y"
        escalate = ask("escalate [y/n]: ", ["y", "n"]).lower() == "y"
        reason = ask("escalate reason: ") if escalate else ""
        difficulty = ask("difficulty [1 easy, 2 medium, 3 hard]: ", ["1", "2", "3"])
        notes = ask("notes (Enter to skip): ")
        label = Label(case_id=case_id, intent=intents[int(number) - 1], compromised=compromised,
                      escalate=escalate, escalate_reason=reason, difficulty=int(difficulty), notes=notes,
                      labelled_at=datetime.now(timezone.utc).isoformat())
        with GOLDEN_JSONL.open("a", encoding="utf-8", newline="\n") as handle:  # saved before the next case
            handle.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
        done.add(case_id)
        counts[label.intent] += 1
    print(f"\nAll {len(pool_ids)} cases labelled: {dict(counts.most_common())}")


if __name__ == "__main__":
    main()
