"""Override statistics for the golden labels (python -m eval.label_stats). Makes no model call.

Prints the override rate on assisted cases, overall and per intent, and phi3's agreement with the blind cases,
then rewrites the "Model-assisted labelling" section of data/golden/labeling_notes.md between its markers.
"""

import json
from collections import Counter
from pathlib import Path

from src import classify

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_JSONL = REPO_ROOT / "data" / "golden" / "golden_set.jsonl"
NOTES_MD = REPO_ROOT / "data" / "golden" / "labeling_notes.md"
START = "<!-- label_stats: start -->"
END = "<!-- label_stats: end -->"
DISCLOSURE = (
    "The golden labels are MODEL-ASSISTED. Every case was shown to me and labelled by me. On assisted cases, "
    "phi3 (`phi3:3.8b-mini-128k-instruct-q4_0` with the compact classifier prompt: the same model and prompt "
    "family as the classify stage) proposed an intent, and I confirmed or corrected it. On blind cases, every "
    "5th pool position, I labelled with the suggestion hidden. The override rate measures how often I disagreed "
    "with a shown suggestion. It is a lower bound on independent judgement, because accepting a shown "
    "suggestion can be anchoring as well as agreement; phi3's agreement on the blind cases, against its "
    "agreement on the assisted ones, measures that anchoring."
)


def share(part: int, whole: int) -> str:
    """'part of whole (x.x%)', or 'none yet' when whole is 0."""
    if whole == 0:
        return "none yet"
    return f"{part} of {whole} ({part / whole:.1%})"


def render(labels: list[dict], order: list[str]) -> str:
    """The notes section: the disclosure, the overall figures and the per-intent override table."""
    assisted = [label for label in labels if label.get("mode") == "assisted"]
    blind = [label for label in labels if label.get("mode") == "blind"]
    manual = len(labels) - len(assisted) - len(blind)
    overridden = [label for label in assisted if not label["accepted"]]
    blind_agreed = sum(1 for label in blind if label["intent"] == label["proposed_intent"])
    lines = [START, "## Model-assisted labelling", "", DISCLOSURE, "",
             f"- Labelled: {len(labels)} ({len(assisted)} assisted, {len(blind)} blind, "
             f"{manual} manual, labelled before --assist existed)",
             f"- **Override rate on assisted cases: {share(len(overridden), len(assisted))}**",
             f"- phi3 agreement with my labels: assisted {share(len(assisted) - len(overridden), len(assisted))}, "
             f"blind {share(blind_agreed, len(blind))}",
             "",
             "Per intent, on assisted cases. By suggestion: how often a suggestion of that intent was overridden. "
             "By label: how often my label of that intent differed from the suggestion.",
             "",
             "| intent | suggested | override rate by suggestion | labelled | override rate by label |",
             "|---|---|---|---|---|"]
    suggested = Counter(label["proposed_intent"] for label in assisted)
    suggested_overridden = Counter(label["proposed_intent"] for label in overridden)
    labelled = Counter(label["intent"] for label in assisted)
    labelled_overridden = Counter(label["intent"] for label in overridden)
    for intent in order:
        lines.append(f"| {intent} | {suggested[intent]} | {share(suggested_overridden[intent], suggested[intent])} "
                     f"| {labelled[intent]} | {share(labelled_overridden[intent], labelled[intent])} |")
    lines.append(END)
    return "\n".join(lines) + "\n"


def main() -> None:
    order, _ = classify.parse_taxonomy(classify.TAXONOMY_MD.read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in GOLDEN_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    section = render(labels, order)
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
