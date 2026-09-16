"""The blind audit's answer guards: keypad cycles are refused, fast answers are warned about. No data, no Ollama."""

import pytest

from src import audit


def test_a_keypad_cycle_is_measured() -> None:
    assert audit.longest_cycle([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 9
    assert audit.longest_cycle([5, 6, 7, 8, 9, 1, 2, 3, 4]) == 9  # the wrap from 9 back to 1 counts
    assert audit.longest_cycle([7, 1, 2, 3, 9, 9]) == 3  # only the run in the middle
    assert audit.longest_cycle([]) == 0
    with pytest.raises(SystemExit):
        audit.refuse_if_cycling([1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_answers_that_look_like_judgements_pass_both_limits() -> None:
    answers = [3, 3, 1, 9, 2, 2, 7, 4, 5]
    assert audit.longest_cycle(answers) < audit.STARTUP_CYCLE
    audit.refuse_if_cycling(answers)  # no exception while the audit runs
    audit.refuse_if_cycling(answers, audit.STARTUP_CYCLE)  # nor when the file is opened again


def test_a_part_cycle_left_on_disk_cannot_be_extended() -> None:
    residue = [1, 2, 3, 4, 5, 6, 7, 8]  # what a refusal at the ninth answer leaves behind
    audit.refuse_if_cycling(residue)  # the live limit alone would let the audit carry on
    with pytest.raises(SystemExit):
        audit.refuse_if_cycling(residue, audit.STARTUP_CYCLE)
    with pytest.raises(SystemExit):  # and the shortest run the startup check refuses
        audit.refuse_if_cycling([9, 1, 2, 3, 4], audit.STARTUP_CYCLE)


def test_the_invalid_first_pass_would_have_been_stopped_at_the_ninth_answer() -> None:
    recorded = [(number % 9) + 1 for number in range(40)]  # the 40 answers of 2026-09-16
    with pytest.raises(SystemExit):
        audit.refuse_if_cycling(recorded[:9])


def test_fast_answers_are_counted_and_warned_about(capsys: pytest.CaptureFixture) -> None:
    assert audit.warn_if_fast(5.0, 0) == 0
    assert capsys.readouterr().out == ""
    assert audit.warn_if_fast(0.9, 2) == 3
    assert "under 2 s" in capsys.readouterr().out
