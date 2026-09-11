"""Pins the drop rules, placeholders and near-duplicate grouping of src/clean.py. Needs no Ollama, so it runs in CI.

tests/fixtures/cases_mini.jsonl holds eight cases in the shape src/ingest.py writes:
  11          a retweet -> dropped
  12          emoji only -> dropped
  13          two content tokens -> dropped
  14          handles, URLs, an email, mojibake and an HTML entity -> normalised in every text field
  15          "Awww...." which the www pattern must leave alone
  92, 71, 85  a chain: 92~71 and 71~85 are near-duplicates, 92 and 85 are not
"""

import json
from pathlib import Path

import pytest

from src import clean

FIXTURE = Path(__file__).parent / "fixtures" / "cases_mini.jsonl"


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> tuple[dict[int, dict], str]:
    """Run the whole stage on the fixture. Returns the cleaned cases by case_id and the printed table."""
    out_path = tmp_path / "cases_clean.jsonl"
    monkeypatch.setattr(clean, "CASES_JSONL", FIXTURE)
    monkeypatch.setattr(clean, "CLEAN_JSONL", out_path)
    clean.main()
    cases = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        cases[case["case_id"]] = case
    return cases, capsys.readouterr().out


def dropped_count(table: str, reason: str) -> int:
    """The number printed on the 'dropped: <reason>' line of the table."""
    for line in table.splitlines():
        if line.startswith("dropped: " + reason):
            return int(line.split()[-1].replace(",", ""))
    raise AssertionError(f"no 'dropped: {reason}' line in the table")


def jaccard(text_a: str, text_b: str) -> float:
    shingles_a, shingles_b = clean.shingles(text_a), clean.shingles(text_b)
    return len(shingles_a & shingles_b) / len(shingles_a | shingles_b)


def test_retweet_emoji_only_and_two_token_messages_are_dropped(run: tuple[dict[int, dict], str]) -> None:
    cases, table = run
    assert sorted(cases) == [14, 15, 71, 85, 92]
    assert dropped_count(table, "retweet") == 1
    assert dropped_count(table, "empty or emoji-only") == 1
    assert dropped_count(table, "under 3 content tokens") == 1


def test_handles_and_urls_become_placeholders(run: tuple[dict[int, dict], str]) -> None:
    case = run[0][14]
    # Also pins: the email keeps its "@", ftfy repairs the mojibake and the &amp;, whitespace is collapsed.
    assert case["customer_text"] == ("@USER @USER my playlist is gone, see URL or URL Email me at jo@example.com"
                                     " - I don't know what happened & I'm stuck")
    assert case["brand_reply"] == "@USER Thanks @USER, try URL ^AB"
    assert case["prior_turns"][0]["text"] == "@USER Hi! Have a look at URL ^AB"


def test_awww_is_not_mangled_by_the_www_pattern(run: tuple[dict[int, dict], str]) -> None:
    assert run[0][15]["customer_text"] == "@USER Awww.... thank you so much for the quick help"


def test_near_duplicate_chain_lands_in_one_group_named_by_its_smallest_id(run: tuple[dict[int, dict], str]) -> None:
    cases, _ = run
    text_a, text_b, text_c = cases[92]["customer_text"], cases[71]["customer_text"], cases[85]["customer_text"]
    # The premise: A~B and B~C clear the threshold but A~C does not, so only the chain through B links A and C.
    assert jaccard(text_a, text_b) >= clean.JACCARD
    assert jaccard(text_b, text_c) >= clean.JACCARD
    assert jaccard(text_a, text_c) < clean.JACCARD
    assert cases[92]["dup_group_id"] == cases[71]["dup_group_id"] == cases[85]["dup_group_id"] == 71
    assert cases[14]["dup_group_id"] is None
    assert cases[15]["dup_group_id"] is None


def test_lang_is_unknown_whenever_confidence_is_below_threshold(run: tuple[dict[int, dict], str]) -> None:
    for case in run[0].values():
        assert 0.0 <= case["lang_confidence"] <= 1.0
        assert case["lang"] == "unknown" or case["lang_confidence"] >= clean.LANG_MIN_CONFIDENCE
    assert clean.detect_language("@USER URL") == ("unknown", 0.0)  # nothing left to detect
