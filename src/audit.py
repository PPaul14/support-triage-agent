"""Golden set, part 4: the blind audit (python -m src.label audit) and the guards on its answers.

The first audit pass was invalid: 40 answers stepping 1..9 through the keypad, entered in 97 seconds
(report/DECISIONS.md, Golden set labels). These guards keep that from reaching the labels a second time.
"""

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.label import AUDIT_JSONL, AUDIT_QUEUE_JSONL, INTENT_KEYS, ask, load_cases, read_jsonl, show

CYCLE_LENGTH = 9  # answers stepping 1, 2, ... 9 in order, 9 wrapping back to 1: a keypad test, not labelling
STARTUP_CYCLE = 5  # in a file already on disk, even a part-cycle this long means the answers were keypresses
FAST_ANSWER_S = 2.0  # under this, the case cannot have been read


@dataclass
class AuditLabel:
    case_id: int
    audit_intent: str  # my blind label, compared with the model label by eval/label_stats.py
    audited_at: str  # ISO 8601, UTC
    seconds: float  # how long the answer took, so a later reader can see whether the case was read


def longest_cycle(numbers: list[int]) -> int:
    """The longest run of answers that steps straight through the keypad, 9 wrapping back to 1."""
    longest = run = 1 if numbers else 0
    for earlier, later in zip(numbers, numbers[1:]):
        if later == earlier % CYCLE_LENGTH + 1:
            run += 1
        else:
            run = 1
        longest = max(longest, run)
    return longest


def refuse_if_cycling(numbers: list[int], limit: int = CYCLE_LENGTH) -> None:
    """Stop the audit when the answers step through the keypad instead of judging the cases. A file already on
    disk is held to the stricter STARTUP_CYCLE, so a part-cycle left by an earlier refusal cannot be extended."""
    run = longest_cycle(numbers)
    if run >= limit:
        raise SystemExit(f"{run} answers in a row stepped straight through the keypad (1..{CYCLE_LENGTH}), which is "
                         f"a keypad test rather than labelling. Nothing further was written. Move "
                         f"{AUDIT_JSONL.name} aside and start the audit again.")


def warn_if_fast(seconds: float, fast_so_far: int) -> int:
    """Warn about an answer given faster than the case can be read, and return the running count."""
    if seconds >= FAST_ANSWER_S:
        return fast_so_far
    print(f"  warning: answered in {seconds:.1f} s, under {FAST_ANSWER_S:.0f} s. That is {fast_so_far + 1} so far; "
          "a label is worth only the reading behind it.")
    return fast_so_far + 1


def audit(intents: list[str]) -> None:
    """Label the audit queue blind: the model's label, the stratum, the language tag and the brand reply stay hidden."""
    queue = [record["case_id"] for record in read_jsonl(AUDIT_QUEUE_JSONL)]
    cases = load_cases(set(queue))
    previous = read_jsonl(AUDIT_JSONL)
    done = set(record["case_id"] for record in previous)
    counts = Counter(record["audit_intent"] for record in previous)
    numbers = [intents.index(record["audit_intent"]) + 1 for record in previous]
    refuse_if_cycling(numbers, STARTUP_CYCLE)  # a part-cycle left on disk must not be extended
    fast = 0
    print(f"{len(done)}/{len(queue)} audited. Blind: the model's label is never shown. Type q to stop.")
    for case_id in queue:
        if case_id in done:
            continue
        show(cases[case_id], len(done) + 1, len(queue), intents, counts, INTENT_KEYS)
        shown_at = time.perf_counter()
        number = ask(f"intent [1-{len(intents)}]: ", [str(n) for n in range(1, len(intents) + 1)])
        seconds = time.perf_counter() - shown_at
        numbers.append(int(number))
        refuse_if_cycling(numbers)  # checked before the answer is written, so a full cycle never reaches the file
        fast = warn_if_fast(seconds, fast)
        record = AuditLabel(case_id=case_id, audit_intent=intents[int(number) - 1],
                            audited_at=datetime.now(timezone.utc).isoformat(), seconds=round(seconds, 1))
        with AUDIT_JSONL.open("a", encoding="utf-8", newline="\n") as handle:  # saved before the next case
            handle.write(json.dumps(asdict(record)) + "\n")
        done.add(case_id)
        counts[record.audit_intent] += 1
    print("\nAudit complete. Run python -m eval.label_stats for the agreement figures.")
