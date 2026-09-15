"""Audit statistics (python -m eval.label_stats). Makes no model call.

Compares my blind audit labels with the model labels on the same cases: overall agreement with a Wilson 95%
interval, and per-intent agreement where the support allows. Prints the figures, then rewrites the "Human audit"
section of data/golden/labeling_notes.md between its markers.
"""

import math
from collections import Counter

from src import classify
from src.label import AUDIT_JSONL, GOLDEN_JSONL, read_jsonl

NOTES_MD = GOLDEN_JSONL.parent / "labeling_notes.md"
START = "<!-- audit_stats: start -->"
END = "<!-- audit_stats: end -->"
AUDIT_SIZE = 40
MIN_SUPPORT = 5  # per-intent agreement is reported only for intents with at least this many audited cases
Z = 1.96  # two-sided 95%


def wilson(agreed: int, total: int) -> tuple[float, float]:
    """Wilson score 95% interval for `agreed` agreements out of `total` audited cases."""
    rate = agreed / total
    denominator = 1 + Z * Z / total
    centre = (rate + Z * Z / (2 * total)) / denominator
    half_width = Z * math.sqrt(rate * (1 - rate) / total + Z * Z / (4 * total * total)) / denominator
    return centre - half_width, centre + half_width


def render(pairs: list[tuple[str, str]], order: list[str]) -> str:
    """The notes section: overall agreement with its interval, then agreement per model-labelled intent."""
    lines = [START, "## Human audit: agreement with the model labels", ""]
    if not pairs:
        lines += [f"Not audited yet: 0 of {AUDIT_SIZE}. Agreement: TBD.", END]
        return "\n".join(lines) + "\n"
    agreed = sum(1 for model, audit in pairs if model == audit)
    low, high = wilson(agreed, len(pairs))
    lines += [f"- Audited: {len(pairs)} of {AUDIT_SIZE}, labelled blind by me.",
              f"- **Agreement with the model labels: {agreed} of {len(pairs)} ({agreed / len(pairs):.1%}), "
              f"Wilson 95% interval {low:.1%} to {high:.1%}.**",
              "",
              f"Per intent, by the model's label; agreement is reported only where at least {MIN_SUPPORT} "
              "audited cases carry that label.",
              "",
              "| model label | audited | agreed | agreement |",
              "|---|---|---|---|"]
    support = Counter(model for model, _ in pairs)
    agreements = Counter(model for model, audit in pairs if model == audit)
    for intent in order:
        if support[intent] >= MIN_SUPPORT:
            share = f"{agreements[intent] / support[intent]:.1%}"
        else:
            share = f"not reported (support below {MIN_SUPPORT})"
        lines.append(f"| {intent} | {support[intent]} | {agreements[intent]} | {share} |")
    lines.append(END)
    return "\n".join(lines) + "\n"


def main() -> None:
    order, _ = classify.parse_taxonomy(classify.TAXONOMY_MD.read_text(encoding="utf-8"))
    model_labels = {}
    for record in read_jsonl(GOLDEN_JSONL):
        if record.get("mode") == "model":
            model_labels[record["case_id"]] = record["intent"]
    pairs = [(model_labels[record["case_id"]], record["audit_intent"]) for record in read_jsonl(AUDIT_JSONL)]
    section = render(pairs, order)
    print(section)
    notes = NOTES_MD.read_text(encoding="utf-8")
    if START in notes:  # replace the section from the last run, so re-running never duplicates it
        notes = notes[:notes.index(START)] + section + notes[notes.index(END) + len(END) + 1:]
    else:
        notes = notes.rstrip("\n") + "\n\n" + section
    NOTES_MD.write_text(notes, encoding="utf-8", newline="\n")
    print(f"updated {NOTES_MD}")


if __name__ == "__main__":
    main()
