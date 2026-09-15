"""Leak prevention for the precedent index, and it must fail loudly: no golden case may retrieve a precedent
from its own thread or near-duplicate group, or one nearly identical to it.

The index tests need data/sample/cases_clean.jsonl and the phi3 cache, both gitignored, so CI skips them and
says why; run them locally. The first two tests read committed files only and run everywhere.
"""

import pytest

from src import baselines, index, retrieve
from src.pipeline import golden_cases

MAX_SIMILARITY = 0.98
needs_data = pytest.mark.skipif(not index.CLEAN_JSONL.exists(),
                                reason="data/sample/cases_clean.jsonl is gitignored: run src.ingest and src.clean")


def test_leak_safe_drops_golden_threads_and_groups() -> None:
    cases = [{"case_id": 1, "thread_id": 10, "dup_group_id": None},
             {"case_id": 2, "thread_id": 20, "dup_group_id": 7},
             {"case_id": 3, "thread_id": 30, "dup_group_id": 8},
             {"case_id": 4, "thread_id": 40, "dup_group_id": None}]
    kept = index.leak_safe(cases, threads={10}, groups={7})
    assert [case["case_id"] for case in kept] == [3, 4]


def test_golden_exclusions_cover_all_150_golden_cases() -> None:
    case_ids, threads, _ = index.golden_exclusions()
    assert len(case_ids) == 150 and len(threads) == 150  # every golden case sits in its own thread


@pytest.fixture(scope="module")
def golden_retrieval() -> tuple[index.Index, list[dict], object]:
    built = index.build()
    cases = golden_cases()  # thread and group ids from cases_clean.jsonl itself, not from the pool file
    return built, cases, retrieve.embed_cases(cases)


def leaks(case: dict, precedent: index.Precedent, similarity: float) -> list[str]:
    found = []
    if precedent.thread_id == case["thread_id"]:
        found.append(f"case {case['case_id']}: precedent {precedent.case_id} is from its thread")
    if case["dup_group_id"] is not None and precedent.dup_group_id == case["dup_group_id"]:
        found.append(f"case {case['case_id']}: precedent {precedent.case_id} is from its duplicate group")
    if similarity >= MAX_SIMILARITY:
        found.append(f"case {case['case_id']}: precedent {precedent.case_id} has cosine {similarity:.4f}")
    return found


@needs_data
def test_index_holds_no_case_from_a_golden_thread_or_group(golden_retrieval) -> None:
    built, cases, _ = golden_retrieval
    threads = set(case["thread_id"] for case in cases)
    groups = set(case["dup_group_id"] for case in cases) - {None}
    leaked = [p.case_id for p in built.precedents if p.thread_id in threads or p.dup_group_id in groups]
    assert not leaked, f"{len(leaked)} precedents share a golden thread or duplicate group: {leaked[:20]}"


@needs_data
def test_top_precedents_of_every_golden_case_are_leak_free(golden_retrieval) -> None:
    built, cases, vectors = golden_retrieval
    assert len(cases) == 150
    problems = []
    for case, vector in zip(cases, vectors):
        for precedent, similarity in index.search(built, vector, retrieve.K):  # any intent: the closest there are
            problems += leaks(case, precedent, similarity)
    assert not problems, f"{len(problems)} leaks:\n" + "\n".join(problems)


@needs_data
def test_retrieval_is_leak_free_under_every_intent_filter(golden_retrieval) -> None:
    built, cases, vectors = golden_retrieval
    intents = sorted(set(precedent.intent for precedent in built.precedents))
    problems = []
    for case, vector in zip(cases, vectors):
        for intent in intents:
            for hit in retrieve.retrieve(built, vector, intent).hits:
                problems += leaks(case, hit.precedent, hit.similarity)
    assert not problems, f"{len(problems)} leaks:\n" + "\n".join(problems)


@needs_data
def test_b1_copies_only_from_the_index_cases(golden_retrieval) -> None:
    built, _, _ = golden_retrieval
    fitted = baselines.fit()
    assert [case["case_id"] for case in fitted.cases] == [precedent.case_id for precedent in built.precedents]
