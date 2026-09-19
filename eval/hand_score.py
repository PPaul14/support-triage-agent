"""Blind hand-scoring of drafted replies (python -m eval.hand_score; REPORT.md 3.4).

Shows a reply with the evidence the judge saw and every system identifier stripped, then asks the same five
binary checks qwen2.5 answered. Each reply is written to data/golden/judge_human_scores.csv as it is finished,
so a stopped session resumes where it stopped. Mechanical input is guarded the way the blind audit is
(src/audit.py): a keypad pattern entered faster than a reply can be read is refused, and fast replies are warned
about. Whether the agent was given past cases is visible in the evidence, so the blinding is partial by
construction; REPORT.md 3.4 says so.
"""

import argparse
import csv
import random
import time
from datetime import datetime, timezone

from eval import data
from eval.judge import CHECKS, JUDGE_MODEL
from src.audit import warn_if_fast
from src.label import GOLDEN_DIR, ask
from src.runs import golden_cases

SCORES_CSV = GOLDEN_DIR / "judge_human_scores.csv"
SAMPLE_SIZE = 30
SEED = 0
FAST_REPLY_S = 2.0
MECHANICAL_RUN = 12  # answers in a row that are all one key, or strictly alternating
PATTERN_REPLIES = 3  # ... and this many replies in a row scored too fast to have been read
FIELDS = ["pair_id", "system", "case_id", "run_id", *CHECKS, "seconds", "scored_at"]


def mechanical(keys: list[str]) -> str | None:
    """Why the last MECHANICAL_RUN answers look like keypresses rather than judgements, or None."""
    recent = keys[-MECHANICAL_RUN:]
    if len(recent) < MECHANICAL_RUN:
        return None
    if len(set(recent)) == 1:
        return f"the last {MECHANICAL_RUN} answers were all {recent[0]!r}"
    if all(later != earlier for earlier, later in zip(recent, recent[1:])):
        return f"the last {MECHANICAL_RUN} answers alternated without a break"
    return None


def refuse_if_mechanical(keys: list[str], reply_seconds: list[float]) -> None:
    """Stop when the answers form a keypad pattern AND arrive too fast to be reading. Either signal alone can be
    honest: five 'no' answers in a row happen, and so does one quick obvious reply."""
    reason = mechanical(keys)
    recent = reply_seconds[-PATTERN_REPLIES:]
    hurried = len(recent) == PATTERN_REPLIES and all(seconds < FAST_REPLY_S for seconds in recent)
    if reason and hurried:
        raise SystemExit(f"Stopping: {reason}, and the last {PATTERN_REPLIES} replies were each scored in under "
                         f"{FAST_REPLY_S:.0f} s. That is keypad input, not scoring. Nothing further was written; "
                         f"move {SCORES_CSV.name} aside and start again.")


def sample_pairs(runs: dict[str, dict[int, dict]], judged: dict) -> list[tuple[str, int]]:
    """SAMPLE_SIZE (system, case_id) pairs drawn from the replies qwen judged, shuffled so the order says nothing."""
    pairs = []
    for (system, _, case_id), record in judged.items():
        if system in runs and case_id in runs[system] and record["usable"]:
            pairs.append((system, case_id))
    pairs.sort()
    rng = random.Random(SEED)
    chosen = rng.sample(pairs, min(SAMPLE_SIZE, len(pairs)))
    rng.shuffle(chosen)
    return chosen


def show(position: int, total: int, case: dict, hits: list[dict], reply: str) -> None:
    """The reply and its evidence, with nothing that names the system that wrote it."""
    print(f"\n{'=' * 78}\nreply {position}/{total}   (case {case['case_id']})")
    if case["prior_turns"]:
        print("\nEarlier in the thread:")
        for turn in case["prior_turns"][-2:]:
            print(f"  [{turn['role']}] {turn['text']}")
    print(f"\nCustomer message:\n  {case['customer_text']}")
    if hits:
        print("\nPast cases the agent was given as evidence:")
        for number, hit in enumerate(hits, start=1):
            print(f"  [{number}] another customer wrote: {hit['precedent']['customer_text']}")
            print(f"      the brand replied:      {hit['precedent']['brand_reply']}")
    print(f"\nTHE REPLY TO SCORE:\n  {reply}\n")
    print("  yes/no on each check. A good reply is NO to the first and YES to the other four.")


def score_reply() -> tuple[dict[str, str], list[str], float]:
    """Ask the five checks. Returns the answers, the keys pressed, and how long the reply took."""
    started = time.perf_counter()
    answers = {}
    keys = []
    for name, (question, _) in CHECKS.items():
        answer = ask(f"  {name}: {question} [y/n]: ", ["y", "n"]).lower()
        answers[name] = "yes" if answer == "y" else "no"
        keys.append(answer)
    return answers, keys, time.perf_counter() - started


def previous_rows() -> list[dict]:
    if not SCORES_CSV.exists():
        return []
    with SCORES_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(row: dict) -> None:
    exists = SCORES_CSV.exists()
    with SCORES_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score drafted replies by hand, blind to the system.")
    parser.add_argument("--model", default=JUDGE_MODEL, help="whose judged replies to sample from")
    args = parser.parse_args()
    selected = data.latest_runs(data.SYSTEMS)
    runs = {system: {trace["case_id"]: trace for trace in traces} for system, traces in selected.items()}
    pairs = sample_pairs(runs, data.judgements(args.model))
    cases = {case["case_id"]: case for case in golden_cases()}
    done = previous_rows()
    scored = set(row["pair_id"] for row in done)
    keys = [row[name][0] for row in done for name in CHECKS]  # the history a resumed session inherits
    reply_seconds = [float(row["seconds"]) for row in done]
    fast = sum(1 for seconds in reply_seconds if seconds < FAST_REPLY_S)
    print(f"{len(scored)}/{len(pairs)} replies scored. Blind: no system is ever named. Type q to stop.")
    for position, (system, case_id) in enumerate(pairs, start=1):
        pair_id = f"{system}:{case_id}"
        if pair_id in scored:
            continue
        trace = runs[system][case_id]
        retrieval = trace.get("retrieval")
        show(position, len(pairs), cases[case_id], [] if retrieval is None else retrieval["hits"],
             trace["output"]["reply"])
        answers, pressed, seconds = score_reply()
        keys += pressed
        reply_seconds.append(seconds)
        refuse_if_mechanical(keys, reply_seconds)  # checked before the row is written
        fast = warn_if_fast(seconds, fast)
        append_row({"pair_id": pair_id, "system": system, "case_id": case_id, "run_id": trace["run_id"],
                    **answers, "seconds": round(seconds, 1),
                    "scored_at": datetime.now(timezone.utc).isoformat()})
        scored.add(pair_id)
    print(f"\nAll {len(pairs)} replies scored. Run python -m eval.agreement for kappa and raw agreement.")


if __name__ == "__main__":
    main()
