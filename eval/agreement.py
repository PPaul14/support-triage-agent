"""Judge validation (REPORT.md 3.4): my blind hand scores against qwen2.5's verdicts on the same replies.

Cohen's kappa and raw agreement for every check, always both: kappa collapses when a check is nearly always the
same answer, and raw agreement alone would overstate agreement there. Where the two raters disagree, the
direction is reported — which way the judge leans — because a judge that is wrong in one direction biases every
quality number in that direction.
"""

import csv
import math
from dataclasses import dataclass

from eval import data
from eval.hand_score import SCORES_CSV
from eval.judge import CHECKS, JUDGE_MODEL

# Landis and Koch (1977), the conventional reading of kappa. Reported as a label, never as a pass mark.
STRENGTH = [(0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"), (0.21, "fair"),
            (0.01, "slight"), (-1.0, "none or worse than chance")]


@dataclass
class Check:
    name: str
    n: int
    agreed: int
    kappa: float
    human_yes: int
    judge_yes: int
    human_yes_judge_no: int
    human_no_judge_yes: int


def strength(value: float) -> str:
    if math.isnan(value):
        return "not defined"
    for threshold, label in STRENGTH:
        if value >= threshold:
            return label
    return "none or worse than chance"


def cohens_kappa(human: list[str], judge: list[str]) -> float:
    """Kappa on two yes/no series. Not defined when chance agreement is already 1, which happens when both
    raters gave the same answer to every reply."""
    total = len(human)
    observed = sum(1 for one, other in zip(human, judge) if one == other) / total
    expected = sum((human.count(answer) / total) * (judge.count(answer) / total) for answer in ("yes", "no"))
    if expected >= 1.0:
        return math.nan
    return (observed - expected) / (1 - expected)


def load_pairs(model: str = JUDGE_MODEL) -> list[tuple[dict, dict]]:
    """(my answers, the judge's answers) for every hand-scored reply the judge also scored usably."""
    judged = data.judgements(model)
    with SCORES_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = []
    for row in rows:
        record = judged.get((row["system"], row["run_id"], int(row["case_id"])))
        if record is not None and record["usable"]:
            pairs.append(({name: row[name] for name in CHECKS}, record["answers"]))
    return pairs


def compare(pairs: list[tuple[dict, dict]]) -> list[Check]:
    checks = []
    for name in CHECKS:
        human = [mine[name] for mine, _ in pairs]
        judge = [theirs[name] for _, theirs in pairs]
        checks.append(Check(
            name=name, n=len(pairs),
            agreed=sum(1 for one, other in zip(human, judge) if one == other),
            kappa=cohens_kappa(human, judge),
            human_yes=human.count("yes"), judge_yes=judge.count("yes"),
            human_yes_judge_no=sum(1 for one, other in zip(human, judge) if one == "yes" and other == "no"),
            human_no_judge_yes=sum(1 for one, other in zip(human, judge) if one == "no" and other == "yes")))
    return checks


def direction(check: Check) -> str:
    """Which way the judge leans against me, in cases, or that it does not lean."""
    if check.human_yes_judge_no == check.human_no_judge_yes:
        return "no lean"
    if check.human_no_judge_yes > check.human_yes_judge_no:
        return f"judge says yes more ({check.human_no_judge_yes} vs {check.human_yes_judge_no})"
    return f"judge says no more ({check.human_yes_judge_no} vs {check.human_no_judge_yes})"


def lines(checks: list[Check]) -> list[str]:
    """The markdown table for REPORT.md 3.4."""
    out = [f"| check | n | raw agreement | Cohen's kappa | my yes | judge yes | direction of disagreement |",
           "|---|---|---|---|---|---|---|"]
    for check in checks:
        value = "not defined" if math.isnan(check.kappa) else f"{check.kappa:.2f} ({strength(check.kappa)})"
        out.append(f"| {check.name} | {check.n} | {check.agreed}/{check.n} "
                   f"({check.agreed / check.n:.0%}) | {value} | {check.human_yes} | {check.judge_yes} | "
                   f"{direction(check)} |")
    return out


def main() -> None:
    pairs = load_pairs()
    if not pairs:
        raise SystemExit(f"no hand scores in {SCORES_CSV}: run python -m eval.hand_score first")
    checks = compare(pairs)
    print(f"Judge validation: {len(pairs)} replies scored by hand, blind, against {JUDGE_MODEL}.\n")
    print("\n".join(lines(checks)))
    headline = next(check for check in checks if check.name == "would_send_unedited")
    print(f"\nwould_send_unedited: {headline.agreed}/{headline.n} agreed "
          f"({headline.agreed / headline.n:.0%}), kappa {headline.kappa:.2f} ({strength(headline.kappa)}).")
    print(f"Every reply and answer: {SCORES_CSV}")


if __name__ == "__main__":
    main()
