"""Golden set, part 2: label every unlabelled pool case with phi3 plus rules (python -m src.model_label).

Makes no model call: the intent is the sampler's cached phi3 estimate, and escalate and compromised follow the
guideline's rules. It then draws the 40-case audit queue that I label blind (python -m src.label audit).
"""

import json
import random
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

from src import classify, golden
from src.label import AUDIT_QUEUE_JSONL, CLEAN_JSONL, GOLDEN_JSONL, POOL_JSONL, REASONS, Label, read_jsonl

AUDIT_SIZE = 40
SEED = 0
# The guideline's "compromised": someone else got into the account. A keyword rule over the customer's own text.
COMPROMISED = re.compile(r"\bhack|hijack|someone else|somebody else|someone (has )?(changed|logged|is using|was using)"
                         r"|not my (device|account)|isn'?t mine|unknown device|without me knowing", re.IGNORECASE)


def rule_fields(case: dict, intent: str) -> tuple[bool, bool, str]:
    """(compromised, escalate, escalate_reason) from the guideline's rules; the first matching reason wins."""
    customer_turns = [turn["text"] for turn in case["prior_turns"] if turn["role"] == "customer"]
    compromised = COMPROMISED.search(" ".join(customer_turns + [case["customer_text"]])) is not None
    if compromised:
        return True, True, REASONS["2"]
    if intent == "billing_subscription":
        return False, True, REASONS["1"]
    if intent == "other_unclear":
        return False, True, REASONS["3"]
    if intent == "followup_diagnostic" and not case["prior_turns"]:
        return False, True, "cannot interpret without earlier turns"
    return False, False, ""


def main() -> None:
    taxonomy = classify.TAXONOMY_MD.read_text(encoding="utf-8")
    order, _ = classify.parse_taxonomy(taxonomy)
    pool_ids = [record["case_id"] for record in read_jsonl(POOL_JSONL)]
    labelled = set(record["case_id"] for record in read_jsonl(GOLDEN_JSONL))
    todo = [case_id for case_id in pool_ids if case_id not in labelled]  # pool order
    if not todo:
        raise SystemExit("Every pool case already has a label: nothing written, audit queue untouched.")
    wanted = set(todo)
    cases = {case["case_id"]: case for case in read_jsonl(CLEAN_JSONL) if case["case_id"] in wanted}
    estimates = golden.cached_estimates([cases[case_id] for case_id in todo], classify.render_block(taxonomy), order)
    missing = [case_id for case_id in todo if case_id not in estimates]
    if missing:  # never call the model here: stop and say which cases lack a cached estimate
        raise SystemExit(f"{len(missing)} cases have no cached phi3 estimate, so nothing was written: {missing}")
    labelled_at = datetime.now(timezone.utc).isoformat()
    labels = []
    for case_id in todo:
        compromised, escalate, reason = rule_fields(cases[case_id], estimates[case_id])
        labels.append(Label(case_id=case_id, intent=estimates[case_id], compromised=compromised, escalate=escalate,
                            escalate_reason=reason, difficulty=None, notes="", labelled_at=labelled_at, mode="model"))
    with GOLDEN_JSONL.open("a", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
    audit_ids = random.Random(SEED).sample(sorted(todo), AUDIT_SIZE)
    with AUDIT_QUEUE_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for case_id in audit_ids:
            handle.write(json.dumps({"case_id": case_id}) + "\n")
    intents = Counter(label.intent for label in labels)
    reasons = Counter(label.escalate_reason for label in labels if label.escalate)
    print(f"model-labelled {len(labels)} cases, no model calls: {dict(intents.most_common())}")
    print(f"escalate: {sum(1 for label in labels if label.escalate)} | "
          f"compromised: {sum(1 for label in labels if label.compromised)} | reasons: {dict(reasons.most_common())}")
    print(f"audit queue: {len(audit_ids)} of the {len(todo)} model-labelled cases, seed {SEED}: {AUDIT_QUEUE_JSONL}")


if __name__ == "__main__":
    main()
