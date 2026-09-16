"""Golden set, part 3: the labelling CLI. `python -m src.label` labels pool cases by hand, and
`python -m src.label audit` runs the blind audit from src/audit.py. Every answer is saved at once."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import classify

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_JSONL = REPO_ROOT / "data" / "sample" / "cases_clean.jsonl"
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
POOL_JSONL = GOLDEN_DIR / "golden_pool.jsonl"
GOLDEN_JSONL = GOLDEN_DIR / "golden_set.jsonl"
AUDIT_QUEUE_JSONL = GOLDEN_DIR / "audit_queue.jsonl"
AUDIT_JSONL = GOLDEN_DIR / "audit_labels.jsonl"
KEY_BLOCK = ("  1 billing  2 login  3 playback  4 playlists/downloads  5 not available\n"
             "  6 feature request  7 reply, problem not named  8 thanks/chatter  9 unclear\n"
             "  compromised: n unless someone else got into their account\n"
             "  escalate: y for 1, 2 with hacking, 9, and 7 without enough context. n otherwise.\n"
             "  difficulty: 1 obvious  2 paused  3 torn        notes: Enter to skip")
INTENT_KEYS = "\n".join(KEY_BLOCK.splitlines()[:2])  # the audit asks for the intent only
REASONS = {"1": "needs account data", "2": "account compromised", "3": "cannot determine intent"}


@dataclass
class Label:
    case_id: int
    intent: str
    compromised: bool  # someone else got into the account: the SEVERE harm rule reads it as data
    escalate: bool
    escalate_reason: str  # empty when escalate is False
    difficulty: int | None  # 1 easy, 2 medium, 3 hard; None on model labels
    notes: str
    labelled_at: str  # ISO 8601, UTC
    mode: str = "manual"  # "manual" (labelled by me here) or "model" (phi3 plus rules, src/model_label.py)


def ask(prompt: str, allowed: list[str] | None = None) -> str:
    """Read one answer, repeating until it is in `allowed` when given. 'q' quits; everything so far is saved."""
    while True:
        answer = input(prompt).strip()
        if answer.lower() == "q":
            raise SystemExit("Stopped. Run the same command again to resume where you left off.")
        if allowed is None or answer.lower() in allowed:
            return answer
        print(f"  Please answer one of: {', '.join(allowed)}")


def show(case: dict, position: int, total: int, intents: list[str], counts: Counter[str], key: str) -> None:
    """Print the case the way the labeller sees it: prior turns and the customer message, nothing else."""
    print(f"\n{'=' * 72}\ncase {position}/{total}   (case_id {case['case_id']})\n{key}")
    if case["prior_turns"]:
        print("\nEarlier in the thread:")
        for turn in case["prior_turns"]:
            print(f"  [{turn['role']}] {turn['text']}")
    print(f"\nCustomer message:\n  {case['customer_text']}\n")
    for number, intent in enumerate(intents, start=1):
        print(f"  {number}  {intent:22} ({counts[intent]} so far)")


def read_jsonl(path: Path) -> list[dict]:
    """Every record in a JSONL file, or an empty list if the file does not exist yet."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cases(case_ids: set[int]) -> dict[int, dict]:
    """The cases_clean records for these case ids."""
    return {case["case_id"]: case for case in read_jsonl(CLEAN_JSONL) if case["case_id"] in case_ids}


def label_by_hand(intents: list[str]) -> None:
    """Label by hand every pool case that has no label yet, in pool order."""
    pool_ids = [record["case_id"] for record in read_jsonl(POOL_JSONL)]
    cases = load_cases(set(pool_ids))
    records = read_jsonl(GOLDEN_JSONL)  # plain dicts: older records carry fields this version no longer writes
    done = set(record["case_id"] for record in records)
    counts = Counter(record["intent"] for record in records)
    print(f"{len(done)}/{len(pool_ids)} already labelled. Type q at any prompt to stop; nothing is lost.\n{KEY_BLOCK}")
    for case_id in pool_ids:
        if case_id in done:
            continue
        show(cases[case_id], len(done) + 1, len(pool_ids), intents, counts, KEY_BLOCK)
        number = ask(f"intent [1-{len(intents)}]: ", [str(n) for n in range(1, len(intents) + 1)])
        compromised = ask("compromised account, someone else got in [y/n]: ", ["y", "n"]).lower() == "y"
        escalate = ask("escalate [y/n]: ", ["y", "n"]).lower() == "y"
        reason = ask("escalate reason [1 account data, 2 compromised, 3 unclear, or type it]: ") if escalate else ""
        reason = REASONS.get(reason, reason)  # a single digit 1-3 is a shortcut; anything else is kept as typed
        difficulty = ask("difficulty [1 easy, 2 medium, 3 hard]: ", ["1", "2", "3"])
        notes = ask("notes (Enter to skip): ")
        label = Label(case_id=case_id, intent=intents[int(number) - 1], compromised=compromised, escalate=escalate,
                      escalate_reason=reason, difficulty=int(difficulty), notes=notes,
                      labelled_at=datetime.now(timezone.utc).isoformat())
        with GOLDEN_JSONL.open("a", encoding="utf-8", newline="\n") as handle:  # saved before the next case
            handle.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
        done.add(case_id)
        counts[label.intent] += 1
    print(f"\nAll {len(pool_ids)} cases labelled: {dict(counts.most_common())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label the golden pool by hand, or audit the model labels blind.")
    parser.add_argument("command", nargs="?", choices=["label", "audit"], default="label")
    command = parser.parse_args().command
    sys.stdout.reconfigure(encoding="utf-8")  # tweets contain emoji; a Windows pipe defaults to cp1252
    intents, _ = classify.parse_taxonomy(classify.TAXONOMY_MD.read_text(encoding="utf-8"))
    if command == "audit":
        from src.audit import audit  # imported here: src/audit.py uses this module's helpers
        audit(intents)
    else:
        label_by_hand(intents)


if __name__ == "__main__":
    main()
