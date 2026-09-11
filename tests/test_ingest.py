"""Pins the fork and drop rules of src/ingest.py on a hand-built CSV. Needs no Ollama, so it runs in CI.

tests/fixtures/twcs_mini.csv holds six tiny threads, rows deliberately out of order:
  101-106  fork: two customers (103, 104) reply to the same brand tweet 102
  201-202  missing parent: 201 replies to tweet 999, which is not in the file
  301-303  two brand replies to 301; the higher id (303) was sent first
  401-406  unanswered customer messages: 403 is followed up by 404, and 406 gets no reply
  601-603  two brand replies to 601 sent in the same second
  701-702  another company's thread (AppleSupport), no SpotifyCares involvement
"""

import json
import sys
from pathlib import Path

import pytest

from src import ingest

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_mini.csv"


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> tuple[dict[int, dict], str]:
    """Run the whole stage on the fixture. Returns the cases by case_id and the printed summary."""
    out_path = tmp_path / "cases.jsonl"
    monkeypatch.setattr(ingest, "RAW_CSV", FIXTURE)
    monkeypatch.setattr(ingest, "CASES_JSONL", out_path)
    monkeypatch.setattr(sys, "argv", ["ingest"])
    ingest.main()
    cases = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        cases[case["case_id"]] = case
    return cases, capsys.readouterr().out


def prior_ids(case: dict) -> list[int]:
    return [turn["tweet_id"] for turn in case["prior_turns"]]


def summary_value(summary: str, label: str) -> str:
    """The text printed after 'label:' in the summary."""
    for line in summary.splitlines():
        if line.strip().startswith(label + ":"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"{label!r} is not in the summary")


def test_exactly_the_expected_cases_are_emitted(run: tuple[dict[int, dict], str]) -> None:
    cases, _ = run
    # Not cases: 201 (missing parent), 403 and 406 (unanswered), 701 (other company), every brand tweet.
    assert sorted(cases) == [101, 103, 104, 301, 401, 404, 601]


def test_fork_each_branch_sees_only_its_own_path(run: tuple[dict[int, dict], str]) -> None:
    cases, _ = run
    assert prior_ids(cases[103]) == [101, 102]
    assert prior_ids(cases[104]) == [101, 102]  # 103 is a sibling, not an ancestor, so 104 never sees it
    assert cases[103]["brand_reply"].startswith("@1001 Thanks.")
    assert cases[104]["brand_reply"].startswith("@1002 Send us a DM")
    assert cases[103]["thread_id"] == cases[104]["thread_id"] == 101
    assert cases[104]["turn_index"] == 2


def test_thread_with_missing_parent_is_dropped(run: tuple[dict[int, dict], str]) -> None:
    cases, summary = run
    assert 201 not in cases
    assert summary_value(summary, "dropped").startswith("1 brand tweets")


def test_earliest_brand_reply_wins(run: tuple[dict[int, dict], str]) -> None:
    cases, summary = run
    assert cases[301]["brand_reply"] == "@1003 1/2 Offline mode needs Premium. ^EF"  # 303: sent first, higher id
    assert cases[601]["brand_reply"] == "@1006 Reply A sent in the same second. ^IJ"  # tie: lower id 602 wins
    assert "earliest reply kept for 2 customer messages" in summary


def test_unanswered_message_is_context_not_a_case(run: tuple[dict[int, dict], str]) -> None:
    cases, _ = run
    assert 403 not in cases
    assert 406 not in cases
    assert prior_ids(cases[404]) == [401, 402, 403]
    assert [turn["role"] for turn in cases[404]["prior_turns"]] == ["customer", "brand", "customer"]


def test_summary_counts(run: tuple[dict[int, dict], str]) -> None:
    _, summary = run
    assert summary_value(summary, "threads") == "4"  # 101, 301, 401, 601: 201 dropped, 701 not the brand's
    assert summary_value(summary, "cases") == "7"
    assert summary_value(summary, "median turns per thread") == "4.5"  # thread sizes 6, 3, 6, 3
    # Only 106 counts: "admin" in 402 must not match "DM".
    assert summary_value(summary, "BRAND REPLIES MENTIONING DM / DIRECT MESSAGE") == "14.3%  (1 of 7)"


def test_timestamps_become_iso_8601(run: tuple[dict[int, dict], str]) -> None:
    cases, _ = run
    assert cases[103]["created_at"] == "2017-10-31T10:02:00+00:00"


def test_find_root_raises_on_a_reply_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        ingest.find_root(1, {1: 2, 2: 1})
