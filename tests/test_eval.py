"""Evaluation arithmetic and the offline escalation re-scoring, on made-up inputs. Needs no data or Ollama."""

import math

import numpy as np
import pytest

from eval import agreement, escalation, hand_score, metrics, reference


def test_f1_and_macro_f1_by_hand() -> None:
    truth = ["a", "a", "b", "b"]
    predicted = ["a", "b", "b", "b"]
    table = metrics.f1_by_class(truth, predicted, ["a", "b", "c"])
    assert table["a"]["f1"] == pytest.approx(2 / 3)  # tp 1, fn 1
    assert table["b"]["f1"] == pytest.approx(0.8)  # tp 2, fp 1
    assert metrics.macro_f1(truth, predicted, ["a", "b", "c"]) == pytest.approx((2 / 3 + 0.8) / 2)  # c: no support


def test_confusion_rows_are_the_truth() -> None:
    assert metrics.confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b"]) == [[1, 1], [0, 1]]


def test_reweighting_matches_the_population_and_equal_weights_change_nothing() -> None:
    truth = ["a", "a", "a", "b"]
    weights = metrics.reweight(truth, {"a": 0.5, "b": 0.5})
    assert weights == pytest.approx([2 / 3, 2 / 3, 2 / 3, 2.0])
    predicted = ["a", "b", "a", "b"]
    assert metrics.macro_f1(truth, predicted, ["a", "b"], [1.0] * 4) == pytest.approx(
        metrics.macro_f1(truth, predicted, ["a", "b"]))


def test_bootstrap_is_deterministic_and_a_system_minus_itself_is_zero() -> None:
    vector = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    statistic = metrics.mean_statistic(vector)
    low, high = metrics.bootstrap_ci(5, statistic)
    assert (low, high) == metrics.bootstrap_ci(5, statistic)
    assert low <= 0.6 <= high
    assert metrics.paired_difference(5, statistic, statistic) == (0.0, 0.0, 0.0)


def make_trace(case_id: int, confidence: float = 0.9, max_sim: float | None = 0.8, l1: object = None,
               l2: object = None, l4: bool = False, accepted: bool = True) -> dict:
    trace = {"case_id": case_id, "l1": l1, "l2": l2, "prediction": {"confidence": confidence},
             "retrieval": None if max_sim is None else {"max_sim": max_sim}, "draft": {"accepted": accepted},
             "judgement": None if (l1 or l2) else {"decision": {"escalate": l4}}}
    trace["output"] = {"escalate": escalation.decide(trace, 0.5, None if max_sim is None else 0.5)}
    return trace


def test_decide_follows_the_ladder() -> None:
    assert escalation.decide(make_trace(1, l1={"code": "fraud"}), 0.5, 0.5)  # L1, whatever comes after
    assert escalation.decide(make_trace(2, confidence=0.4), 0.5, 0.5)  # L3: low confidence
    assert escalation.decide(make_trace(3, accepted=False), 0.5, 0.5)  # L3: the draft failed its checks
    close_enough = make_trace(4, max_sim=0.55)
    assert not escalation.decide(close_enough, 0.5, 0.5)  # passes L3, and L4 says auto
    assert escalation.decide(close_enough, 0.5, 0.6)  # the similarity gate raised above it
    assert not escalation.decide(make_trace(5, max_sim=None), 0.5, 0.9)  # no-RAG: no similarity gate
    assert escalation.decide(make_trace(6, l4=True), 0.5, 0.5)  # L4 says escalate
    unasked = make_trace(7)
    unasked["judgement"] = None
    with pytest.raises(RuntimeError):
        escalation.decide(unasked, 0.5, 0.5)


def test_harm_is_severe_for_billing_and_compromised_accounts_only() -> None:
    labels = [{"escalate": True, "intent": "billing_subscription", "compromised": False},
              {"escalate": True, "intent": "account_access", "compromised": True},
              {"escalate": True, "intent": "other_unclear", "compromised": False},
              {"escalate": False, "intent": "playback_failure", "compromised": False}]
    severe, ordinary = escalation.harms([False, False, False, False], labels)
    assert severe == [1, 1, 0, 0] and ordinary == [0, 0, 1, 0]
    assert escalation.harms([True, True, True, True], labels) == ([0, 0, 0, 0], [0, 0, 0, 0])


def test_operating_point_respects_the_budget() -> None:
    points = [escalation.Point(0.5, 0.5, [], 100, 1, 0),  # most auto-handled, but one SEVERE
              escalation.Point(0.6, 0.5, [], 90, 0, 8),  # 8 ORDINARY is over 7 of 150
              escalation.Point(0.7, 0.5, [], 80, 0, 7),
              escalation.Point(0.8, 0.5, [], 70, 0, 2)]
    assert escalation.operating_point(points, 150) is points[2]
    assert escalation.operating_point(points[:2], 150) is None


def test_sweep_covers_the_grid_and_refuses_decisions_it_cannot_reproduce() -> None:
    labels = [{"escalate": False, "intent": "playback_failure", "compromised": False}] * 3
    with_retrieval = [make_trace(case_id) for case_id in range(3)]
    assert len(escalation.sweep(with_retrieval, labels)) == len(escalation.CONFIDENCE_GRID) * len(
        escalation.SIMILARITY_GRID)
    no_rag = [make_trace(case_id, max_sim=None) for case_id in range(3)]
    assert len(escalation.sweep(no_rag, labels)) == len(escalation.CONFIDENCE_GRID)
    with_retrieval[1]["output"]["escalate"] = True  # a recorded decision the ladder would not have made
    with pytest.raises(RuntimeError):
        escalation.sweep(with_retrieval, labels)


def test_rouge_l_and_reference_tidying() -> None:
    assert reference.rouge_l("we can help with that", "we can help with that") == pytest.approx(1.0)
    assert reference.rouge_l("we can help with that", "entirely different wording here") == 0.0
    assert reference.rouge_l("", "anything") == 0.0
    # 3 of 4 reference words in order, 3 of 3 candidate words: precision 1.0, recall 0.75
    assert reference.rouge_l("log out and back in", "log out and in") == pytest.approx(2 * 1.0 * 0.8 / 1.8)
    assert reference.tidy_reference("@USER @USER Try logging out. /KL") == "Try logging out."


def test_a_keypad_pattern_is_refused_only_when_it_is_also_too_fast() -> None:
    same = ["y"] * hand_score.MECHANICAL_RUN
    alternating = ["y", "n"] * (hand_score.MECHANICAL_RUN // 2)
    judged = ["y", "n", "n", "y", "n", "y", "y", "n", "n", "n", "y", "n"]
    assert hand_score.mechanical(same)
    assert hand_score.mechanical(alternating)
    assert hand_score.mechanical(judged) is None  # a real pattern of answers is neither
    assert hand_score.mechanical(same[:-1]) is None  # too few answers to judge
    hand_score.refuse_if_mechanical(same, [30.0, 25.0, 40.0])  # a pattern, but read at human speed
    hand_score.refuse_if_mechanical(judged, [0.3, 0.2, 0.1])  # fast, but not a pattern
    with pytest.raises(SystemExit):
        hand_score.refuse_if_mechanical(same, [0.3, 0.2, 0.1])


def test_cohens_kappa_against_hand_worked_cases() -> None:
    assert agreement.cohens_kappa(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"]) == pytest.approx(1.0)
    assert agreement.cohens_kappa(["yes", "no"], ["no", "yes"]) == pytest.approx(-1.0)
    # po 0.5 and pe 0.5: agreement exactly at chance
    assert agreement.cohens_kappa(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"]) == pytest.approx(0.0)
    # both raters said yes to everything: chance agreement is already 1, so kappa says nothing
    assert math.isnan(agreement.cohens_kappa(["yes"] * 4, ["yes"] * 4))
    assert agreement.strength(float("nan")) == "not defined"
    assert agreement.strength(0.75) == "substantial"
