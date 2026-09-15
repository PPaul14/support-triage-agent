"""Stage 7 (escalate): the four-layer ladder. Layers are read in order and the first that fires decides.

L1  deterministic regex on the customer's own words. It needs no model output, and no later layer can
    override it.
L2  the intent policy from docs/taxonomy.md: intents that never auto-handle.
L3  gates on the pipeline's own numbers: intent confidence, precedent similarity, and a draft that failed
    its checks twice.
L4  phi3's judgement, constrained to a fixed reason_code enum. Asked only where L1 and L2 did not fire.
"""

import hashlib
import re
from dataclasses import dataclass

from src import llm
from src.classify import CLASSIFY_MODEL
from src.model_label import COMPROMISED

CONFIDENCE_THRESHOLD = 0.50  # a default, not tuned: the operating curve (REPORT.md 3.2) sweeps it
L1_TRIGGERS = {  # checked in this order; the first match is the reason given
    "fraud": re.compile(r"\bfraud\w*|\bscam(?:s|med|mer|mers)?\b|\bstolen card\b|\bcard (?:was |got )?stolen\b",
                        re.IGNORECASE),
    "chargeback": re.compile(r"\bcharge ?backs?\b|\bdispute (?:the|this|a|my) (?:charge|payment|transaction)s?\b"
                             r"|\breverse (?:the|this) (?:charge|payment)\b", re.IGNORECASE),
    "legal_threat": re.compile(r"\b(?:lawyers?|attorneys?|solicitors?|lawsuit|legal action|small claims|class action"
                               r"|trading standards|ombudsman)\b|\bsu(?:e|ing) (?:you|spotify|u)\b"
                               r"|\btake (?:you|spotify) to court\b", re.IGNORECASE),
    "account_compromise": COMPROMISED,  # the same rule that sets the golden labels' compromised field
    "unauthorised_charge": re.compile(r"\bunauthori[sz]ed\b|\b(?:didn'?t|did not|never) authori[sz]e"
                                      r"|\bwithout my (?:permission|consent|knowledge|authori[sz]ation)\b",
                                      re.IGNORECASE),
    "data_deletion": re.compile(r"\bgdpr\b|\bdata protection\b|\bright to be forgotten\b|\b(?:delete|erase|remove|wipe)"
                                r" (?:all )?(?:of )?my (?:personal )?(?:data|information|info|details|account)\b",
                                re.IGNORECASE),
    "self_harm": re.compile(r"\bsuicid\w*|\bkill(?:ing)? myself\b|\bend(?:ing)? my life\b|\bself[- ]?harm\w*"
                            r"|\bhurt(?:ing)? myself\b|\bwant to die\b", re.IGNORECASE),
    "press": re.compile(r"\bjournalists?\b|\breporters?\b|\b(?:press|media) (?:inquiry|enquiry|request|office)\b"
                        r"|\bwriting (?:a|an) (?:story|article|piece)\b|\bcomment for (?:a|an|our|my) (?:story|article)\b",
                        re.IGNORECASE),
    "abuse": re.compile(r"\bharass\w*|\babus(?:e|ed|ive|ing)\b|\bthreat(?:s|en|ened|ening)?\b|\bstalk(?:er|ing|ed)?\b"
                        r"|\bracis[tm]\b|\bhomophob\w*|\bsexis[tm]\b|\bhate speech\b", re.IGNORECASE),
}
ALWAYS_ESCALATE = {  # the "escalate, always" rows of the handling column in docs/taxonomy.md
    "billing_subscription": "billing_subscription always escalates: it needs account and payment data",
    "other_unclear": "other_unclear always escalates: the agent cannot tell what is needed",
}
L4_REASONS = {
    "none": "phi3 judged that the agent can answer this on its own",
    "needs_account_data": "phi3 judged that answering needs this customer's account or payment data",
    "needs_staff_action": "phi3 judged that the customer needs something only staff can do",
    "unclear_request": "phi3 could not tell what the customer needs",
}
L4_PROMPT = ("You triage tweets sent to Spotify's customer support. An automatic agent can answer with general "
             "information and standard troubleshooting steps. It cannot see accounts or payments, and it cannot "
             "change anything. Decide whether the agent can answer the customer message below on its own, or "
             "must hand it to a human. Choose one reason code:\n"
             "none: the agent can answer it on its own.\n"
             "needs_account_data: answering needs this customer's account, payment or order details.\n"
             "needs_staff_action: the customer needs something only staff can do, such as changing or restoring "
             "their account.\n"
             "unclear_request: you cannot tell what the customer needs.\n"
             'Reply with JSON only, in the form {"reason_code": "<code>"}.\n\n')


@dataclass
class Decision:
    escalate: bool
    reason_code: str
    reason_text: str
    layer: int | None  # 1 to 4; None for the baselines, which have no ladder


@dataclass
class Judgement:
    decision: Decision
    prompt_sha256: str
    response: llm.LLMResponse


def customer_words(case: dict) -> str:
    """The customer's own text: earlier customer turns and the message. Brand turns are never matched."""
    turns = [turn["text"] for turn in case["prior_turns"] if turn["role"] == "customer"]
    return " ".join(turns + [case["customer_text"]])


def layer1(case: dict) -> Decision | None:
    text = customer_words(case)
    for code, pattern in L1_TRIGGERS.items():
        match = pattern.search(text)
        if match is not None:
            return Decision(True, code, f"L1 {code} rule matched the customer's words: \"{match.group(0)}\"", 1)
    return None


def layer2(case: dict, intent: str) -> Decision | None:
    if intent in ALWAYS_ESCALATE:
        return Decision(True, f"policy_{intent}", ALWAYS_ESCALATE[intent], 2)
    if intent == "followup_diagnostic" and not case["prior_turns"]:  # the guideline's split rule for this intent
        return Decision(True, "policy_followup_without_context",
                        "followup_diagnostic with no earlier turns: nothing to continue from", 2)
    return None


def layer3(confidence: float, max_sim: float | None, floor: float, draft_ok: bool,
           threshold: float = CONFIDENCE_THRESHOLD) -> Decision | None:
    """max_sim is None in the no-RAG ablation, which retrieves nothing and so has no similarity gate."""
    if confidence < threshold:
        return Decision(True, "low_intent_confidence", f"intent confidence {confidence:.3f} is below {threshold}", 3)
    if max_sim is not None and max_sim < floor:
        return Decision(True, "no_close_precedent", f"best precedent similarity {max_sim:.3f} is below {floor}", 3)
    if not draft_ok:
        return Decision(True, "draft_failed_checks", "the draft failed its checks twice (src/generate.py)", 3)
    return None


def judge(case: dict, intent: str) -> Judgement:
    """Layer 4: phi3 picks one reason code; "none" means auto-handle."""
    lines = [f"Predicted intent: {intent}"]
    lines += [f"Earlier {turn['role']} message: {turn['text']}" for turn in case["prior_turns"][-2:]]
    lines.append(f"Customer message: {case['customer_text']}")
    prompt = L4_PROMPT + "\n".join(lines)
    schema = {"type": "object", "properties": {"reason_code": {"type": "string", "enum": list(L4_REASONS)}},
              "required": ["reason_code"]}
    response = llm.complete(prompt, CLASSIFY_MODEL, schema=schema, num_predict=32)
    code = response.parsed["reason_code"]
    decision = Decision(code != "none", code, L4_REASONS[code], 4)
    return Judgement(decision, hashlib.sha256(prompt.encode("utf-8")).hexdigest(), response)


def decide(l1: Decision | None, l2: Decision | None, l3: Decision | None, l4: Decision | None) -> Decision:
    """The first layer that fired, in order L1, L2, L3, L4: a later layer can never override an earlier one."""
    for decision in (l1, l2, l3):
        if decision is not None:
            return decision
    if l4 is None:
        raise ValueError("L1 to L3 did not fire and there is no L4 judgement to fall back on")
    return l4
