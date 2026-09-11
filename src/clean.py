"""Stage 2: normalise case text, drop unusable messages, tag language and flag near-duplicates."""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import ftfy
from datasketch import MinHash, MinHashLSH
from langdetect import DetectorFactory, LangDetectException, detect_langs

from src.ingest import Case, Turn

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_JSONL = REPO_ROOT / "data" / "sample" / "cases.jsonl"
CLEAN_JSONL = REPO_ROOT / "data" / "sample" / "cases_clean.jsonl"
URL = re.compile(r"https?://\S+|\bwww\.\S+")  # \b: the "www." inside "awww...." is not a URL
HANDLE = re.compile(r"(?<!\w)@\w+")  # the lookbehind leaves the "@" inside an email address alone
PLACEHOLDERS = {"@USER", "URL"}
MIN_CONTENT_TOKENS = 3
LANG_MIN_CONFIDENCE = 0.90
JACCARD = 0.85
NUM_PERM = 128  # MinHash permutations; datasketch's fixed default seed keeps signatures reproducible
SHINGLE = 5  # characters per shingle
DetectorFactory.seed = 0  # langdetect samples randomly; a fixed seed makes `lang` reproducible


@dataclass
class CleanCase(Case):
    # `lang` is a search aid for finding non-English cases when building the golden set. Never use it as a
    # filter: short English tweets come back as "af" or "so" even at 1.0 confidence ("Nope still doesn't work :(").
    lang: str  # langdetect code, or "unknown" below LANG_MIN_CONFIDENCE or when there is nothing to detect
    lang_confidence: float  # probability of langdetect's top guess, kept even when that guess is discarded
    dup_group_id: int | None  # smallest case_id among its near-duplicates; None if it has none


def normalise(text: str) -> str:
    """Repair with ftfy, swap URLs then handles for placeholders (a URL can contain "@"), collapse whitespace."""
    text = ftfy.fix_text(text)
    text = URL.sub("URL", text)
    text = HANDLE.sub("@USER", text)
    return " ".join(text.split())


def drop_reason(raw_text: str, text: str) -> str | None:
    """Why a customer message is unusable, or None if it is kept. Placeholders do not count as content."""
    if raw_text.lstrip().startswith("RT @"):
        return "retweet"
    words = [token for token in text.split() if token not in PLACEHOLDERS]
    if not any(character.isalnum() for character in "".join(words)):
        return "empty or emoji-only"  # no letter or digit left once placeholders are removed
    if len(words) < MIN_CONTENT_TOKENS:
        return f"under {MIN_CONTENT_TOKENS} content tokens"
    return None


def detect_language(text: str) -> tuple[str, float]:
    """Top langdetect guess and its probability, on content words only: placeholders pull tweets towards English."""
    try:
        best = detect_langs(" ".join(token for token in text.split() if token not in PLACEHOLDERS))[0]
    except LangDetectException:  # raised when there are no letters to go on
        return "unknown", 0.0
    lang = best.lang if best.prob >= LANG_MIN_CONFIDENCE else "unknown"
    return lang, round(best.prob, 3)


def shingles(text: str) -> set[str]:
    """Lowercased character 5-grams without @USER (shared handles would make tweets look alike); min one shingle."""
    content = " ".join(token for token in text.lower().split() if token != "@user")
    return set(content[start:start + SHINGLE] for start in range(max(1, len(content) - SHINGLE + 1)))


def find_group(group_of: dict[int, int], case_id: int) -> int:
    """Follow union-find links to the group's root, which is always its smallest case_id."""
    while group_of[case_id] != case_id:
        case_id = group_of[case_id]
    return case_id


def find_duplicate_groups(texts: dict[int, str]) -> dict[int, int]:
    """Map case_id -> smallest case_id of its near-duplicate group. Cases with no near-duplicate are left out."""
    shingle_sets = {case_id: shingles(texts[case_id]) for case_id in sorted(texts)}
    lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
    group_of = {case_id: case_id for case_id in shingle_sets}  # union-find: every id points towards its root
    for case_id, shingle_set in shingle_sets.items():
        signature = MinHash(num_perm=NUM_PERM)
        signature.update_batch([shingle.encode("utf-8") for shingle in sorted(shingle_set)])
        for other_id in lsh.query(signature):  # only earlier cases are indexed yet, so each pair is seen once
            other_set = shingle_sets[other_id]
            if len(shingle_set & other_set) / len(shingle_set | other_set) >= JACCARD:  # LSH proposes, Jaccard decides
                root, other_root = find_group(group_of, case_id), find_group(group_of, other_id)
                group_of[max(root, other_root)] = min(root, other_root)  # the smaller id stays the root
        lsh.insert(case_id, signature)
    roots = {case_id: find_group(group_of, case_id) for case_id in shingle_sets}
    sizes = Counter(roots.values())
    return {case_id: root for case_id, root in roots.items() if sizes[root] > 1}


def print_table(before: list[Case], after: list[CleanCase], dropped: Counter[str]) -> None:
    """Before/after table for customer_text, then the drops, languages and near-duplicate groups."""
    labels = ["cases", "distinct customer texts", "customer texts with a raw URL", "customer texts with an HTML entity"]
    columns = []
    for cases in (before, after):
        texts = [case.customer_text for case in cases]
        columns.append([len(texts), len(set(texts)), sum(1 for text in texts if URL.search(text)),
                        sum(1 for text in texts if re.search(r"&(amp|lt|gt|quot);", text))])
    print(f"{'':36}{'before':>10}{'after':>10}")
    for label, value_before, value_after in zip(labels, columns[0], columns[1]):
        print(f"{label:36}{value_before:>10,}{value_after:>10,}")
    for reason in ("retweet", "empty or emoji-only", f"under {MIN_CONTENT_TOKENS} content tokens"):
        print(f"{'dropped: ' + reason:36}{dropped[reason]:>20,}")
    languages = Counter(case.lang for case in after)
    print("languages: " + ", ".join(f"{lang} {count:,}" for lang, count in languages.most_common(8)))
    group_sizes = Counter(case.dup_group_id for case in after if case.dup_group_id is not None)
    print(f"near-duplicate groups: {len(group_sizes):,}, covering {sum(group_sizes.values()):,} cases "
          f"(largest {max(group_sizes.values(), default=0):,})")


def main() -> None:
    before: list[Case] = []
    with CASES_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record["prior_turns"] = [Turn(**turn) for turn in record["prior_turns"]]
            before.append(Case(**record))
    after: list[CleanCase] = []
    dropped: Counter[str] = Counter()
    for case in before:
        customer_text = normalise(case.customer_text)
        reason = drop_reason(case.customer_text, customer_text)
        if reason is not None:
            dropped[reason] += 1
            continue
        cleaned = replace(case, customer_text=customer_text, brand_reply=normalise(case.brand_reply),
                          prior_turns=[replace(turn, text=normalise(turn.text)) for turn in case.prior_turns])
        lang, lang_confidence = detect_language(customer_text)
        after.append(CleanCase(**vars(cleaned), lang=lang, lang_confidence=lang_confidence, dup_group_id=None))
    groups = find_duplicate_groups({case.case_id: case.customer_text for case in after})
    for case in after:
        case.dup_group_id = groups.get(case.case_id)
    with CLEAN_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for case in after:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print_table(before, after, dropped)


if __name__ == "__main__":
    main()
