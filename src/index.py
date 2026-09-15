"""Stage 5 (retrieve), part 1: the precedent index (python -m src.index prints what it holds).

A precedent is a past case: its customer_text is embedded with all-MiniLM-L6-v2 on CPU (src.taxonomy.embed,
cached on disk) and its payload is the brand_reply. The index leaves out every case that shares a thread_id or
a dup_group_id with any golden case, and every case embedded within MAX_SIMILARITY of one, so no golden case can
be answered from its own conversation or from a near-copy of itself. tests/test_index.py checks all 150.
"""

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import classify, golden, taxonomy
from src.label import CLEAN_JSONL, GOLDEN_JSONL, POOL_JSONL, read_jsonl

# A precedent this close to a golden case is the same message. MinHash can miss these: two short thanks that
# differ only in their emoji share few character 5-grams, while MiniLM embeds them identically.
MAX_SIMILARITY = 0.98


@dataclass
class Precedent:
    case_id: int
    thread_id: int
    dup_group_id: int | None
    intent: str  # estimated (see build), never labelled: retrieval filters on it
    customer_text: str
    brand_reply: str


@dataclass
class Index:
    precedents: list[Precedent]  # sorted by case_id
    vectors: np.ndarray  # one unit-length row per precedent, in the same order


def golden_exclusions() -> tuple[set[int], set[int], set[int]]:
    """The case_ids, thread_ids and dup_group_ids of the golden cases. Fails loudly if the golden files disagree."""
    pool = read_jsonl(POOL_JSONL)
    pool_ids = set(entry["case_id"] for entry in pool)
    labelled_ids = set(record["case_id"] for record in read_jsonl(GOLDEN_JSONL))
    if len(pool_ids) != golden.GOLDEN_SIZE or labelled_ids != pool_ids:
        raise RuntimeError(f"{POOL_JSONL.name} and {GOLDEN_JSONL.name} must hold the same {golden.GOLDEN_SIZE} "
                           f"cases (found {len(pool_ids)} and {len(labelled_ids)}): the leak exclusions depend on it")
    threads = set(entry["thread_id"] for entry in pool)
    groups = set(entry["dup_group_id"] for entry in pool) - {None}
    return pool_ids, threads, groups


def leak_safe(cases: list[dict], threads: set[int], groups: set[int]) -> list[dict]:
    """The cases that share neither a thread_id nor a dup_group_id with the golden set."""
    kept = []
    for case in cases:
        if case["thread_id"] in threads:
            continue
        if case["dup_group_id"] is not None and case["dup_group_id"] in groups:
            continue
        kept.append(case)
    return kept


def embedding_text(text: str) -> str:
    """customer_text without whole-token @USER, as src/taxonomy.py embeds it: a shared handle is not content."""
    return " ".join(token for token in text.split() if token != "@USER")


def precedent_cases() -> tuple[list[dict], np.ndarray]:
    """The cases that may be precedents, sorted by case_id, and their embeddings. B1 copies from the same list."""
    golden_ids, threads, groups = golden_exclusions()
    cases = sorted(read_jsonl(CLEAN_JSONL), key=lambda case: case["case_id"])
    kept = leak_safe(cases, threads, groups)
    vectors = taxonomy.embed([embedding_text(case["customer_text"]) for case in kept])
    golden_vectors = taxonomy.embed([embedding_text(case["customer_text"]) for case in cases
                                     if case["case_id"] in golden_ids])
    closest = (vectors @ golden_vectors.T).max(axis=1)
    rows = [row for row in range(len(kept)) if closest[row] < MAX_SIMILARITY]
    print(f"precedents: {len(rows):,} of {len(cases):,} cases; left out {len(cases) - len(kept)} sharing one of "
          f"{len(threads)} golden threads or {len(groups)} golden duplicate groups, and {len(kept) - len(rows)} "
          f"embedded within {MAX_SIMILARITY} of a golden case")
    return [kept[row] for row in rows], vectors[rows]


def silver_labels() -> dict[int, str]:
    """phi3's cached intent estimates over the sampler's 1,500-case pool, minus golden threads and groups.
    Read from the disk cache exactly as src/golden.py drew them: no model call."""
    _, threads, groups = golden_exclusions()
    taxonomy_text = classify.TAXONOMY_MD.read_text(encoding="utf-8")
    order, _ = classify.parse_taxonomy(taxonomy_text)
    eligible, _, _, _ = golden.load_eligible(taxonomy_text)
    estimate_pool = random.Random(golden.SEED).sample(eligible, golden.ESTIMATE_POOL_SIZE)  # the sampler's first draw
    estimates = golden.cached_estimates(leak_safe(estimate_pool, threads, groups), classify.render_block(taxonomy_text),
                                        order)
    if len(estimates) < golden.GOLDEN_SIZE:
        raise RuntimeError(f"only {len(estimates)} cached phi3 estimates found: is artifacts/llm_cache/ present?")
    return estimates


def build() -> Index:
    """Every precedent case with its embedding and an estimated intent. Precedents have no labels, so a
    logistic regression on the embeddings, trained on phi3's cached estimates, gives them one."""
    kept, vectors = precedent_cases()
    labels = silver_labels()
    row_of = {case["case_id"]: row for row, case in enumerate(kept)}
    train_ids = sorted(case_id for case_id in labels if case_id in row_of)  # training cases are precedents too
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(vectors[[row_of[case_id] for case_id in train_ids]], [labels[case_id] for case_id in train_ids])
    intents = classifier.predict(vectors)
    precedents = []
    for case, intent in zip(kept, intents):
        precedents.append(Precedent(case["case_id"], case["thread_id"], case["dup_group_id"], str(intent),
                                    case["customer_text"], case["brand_reply"]))
    print(f"index: {len(precedents):,} precedents; intents from a classifier trained on {len(train_ids):,} "
          f"cached phi3 estimates")
    return Index(precedents, vectors)


def search(index: Index, query: np.ndarray, k: int, intent: str | None = None) -> list[tuple[Precedent, float]]:
    """The k most similar precedents, best first, optionally of one intent only. Rows and query are unit
    length, so the dot product is the cosine similarity."""
    similarities = index.vectors @ query
    found = []
    for row in np.argsort(-similarities, kind="stable"):
        if intent is None or index.precedents[row].intent == intent:
            found.append((index.precedents[row], float(similarities[row])))
            if len(found) == k:
                break
    return found


def main() -> None:
    built = build()
    counts = Counter(precedent.intent for precedent in built.precedents)
    print("estimated precedent intents: " + ", ".join(f"{name} {count:,}" for name, count in counts.most_common()))


if __name__ == "__main__":
    main()
