"""The escalation ladder's deterministic layers. Every L1 trigger must fire on the customer's words, and no
later layer may override it. Needs no Ollama, so it runs in CI."""

import pytest

from src import escalate

L1_EXAMPLES = {  # made-up messages, one or more per trigger
    "fraud": ["This is fraud, you took money from my account", "spotify is a scam"],
    "chargeback": ["I'm going to file a chargeback with my bank", "I will dispute the charge with my card company"],
    "legal_threat": ["I'm talking to my lawyer about this", "I will sue spotify if this isn't fixed",
                     "reporting you to trading standards"],
    "account_compromise": ["my account has been hacked", "someone else is using my account"],
    "unauthorised_charge": ["there's an unauthorised payment on my card", "I never authorized this charge"],
    "data_deletion": ["please delete my data under GDPR", "I want you to delete my account and everything on it"],
    "self_harm": ["honestly I want to kill myself", "thinking about suicide tbh"],
    "press": ["I'm a journalist writing a story on streaming outages",
              "press enquiry: can you give a comment for our article"],
    "abuse": ["another user keeps harassing me in the comments", "this playlist name is racist"],
}
ORDINARY = ["my songs keep skipping on iOS", "I was charged twice this month, can you help?",
            "please add the new album", "thanks, that fixed it!", "how do I reset my password?",
            "my playlist disappeared after the update", "the ads are way too loud",
            "why was the album removed from my library"]
AUTO = escalate.Decision(False, "none", "L4 says auto-handle", 4)


def make_case(text: str, prior_turns: list[dict] | None = None) -> dict:
    return {"case_id": 1, "thread_id": 1, "customer_text": text, "prior_turns": prior_turns or [],
            "brand_reply": "", "dup_group_id": None}


def test_every_l1_trigger_has_examples() -> None:
    assert set(L1_EXAMPLES) == set(escalate.L1_TRIGGERS)


@pytest.mark.parametrize("code,text", [(code, text) for code, texts in L1_EXAMPLES.items() for text in texts])
def test_l1_fires_and_no_later_layer_overrides_it(code: str, text: str) -> None:
    case = make_case(text)
    first = escalate.layer1(case)
    assert first is not None and first.escalate and first.layer == 1 and first.reason_code == code
    # Everything after L1 says auto-handle: a confident auto intent, a close precedent, a good draft, and L4.
    final = escalate.decide(first, escalate.layer2(case, "chatter_thanks"), escalate.layer3(1.0, 1.0, 0.5, True), AUTO)
    assert final == first


@pytest.mark.parametrize("text", ORDINARY)
def test_l1_stays_quiet_on_ordinary_messages(text: str) -> None:
    assert escalate.layer1(make_case(text)) is None


def test_l1_reads_earlier_customer_turns_but_never_brand_turns() -> None:
    customer_said = make_case("any update?", [{"role": "customer", "text": "my account was hacked"}])
    brand_said = make_case("any update?", [{"role": "brand", "text": "If your account was hacked, reset it."}])
    assert escalate.layer1(customer_said).reason_code == "account_compromise"
    assert escalate.layer1(brand_said) is None


@pytest.mark.parametrize("intent", ["billing_subscription", "other_unclear"])
def test_l2_never_auto_handles_billing_or_unclear(intent: str) -> None:
    decision = escalate.layer2(make_case("hello", [{"role": "brand", "text": "hi"}]), intent)
    assert decision is not None and decision.escalate and decision.layer == 2


def test_l2_escalates_a_followup_only_without_earlier_turns() -> None:
    assert escalate.layer2(make_case("Did not work."), "followup_diagnostic").layer == 2
    assert escalate.layer2(make_case("Did not work.", [{"role": "brand", "text": "Try a restart"}]),
                           "followup_diagnostic") is None
    assert escalate.layer2(make_case("songs won't play"), "playback_failure") is None


def test_l3_gates() -> None:
    assert escalate.layer3(0.2, 0.9, 0.5, True).reason_code == "low_intent_confidence"
    assert escalate.layer3(0.9, 0.3, 0.5, True).reason_code == "no_close_precedent"
    assert escalate.layer3(0.9, 0.9, 0.5, False).reason_code == "draft_failed_checks"
    assert escalate.layer3(0.9, None, 0.5, True) is None  # no-RAG: no similarity gate
    assert escalate.layer3(0.9, 0.9, 0.5, True) is None


def test_decide_falls_back_to_l4_and_refuses_to_guess_without_it() -> None:
    assert escalate.decide(None, None, None, AUTO) == AUTO
    with pytest.raises(ValueError):
        escalate.decide(None, None, None, None)
