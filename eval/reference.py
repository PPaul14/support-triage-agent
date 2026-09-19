"""Reference similarity (REPORT.md 3.6): ROUGE-L between each system's reply and the brand's real reply.

A diagnostic, never a headline. 31.6% of the brand's visible replies move the customer to DM (finding 1.1), so
imitating the reference is not the same as answering well. The reference is stripped of its leading @USER and of
the agent's trailing initials, neither of which any system emits.
"""

import re

from eval import metrics

SIGNATURE = re.compile(r"\s*/[A-Z]{2}\s*$")


def tidy_reference(text: str) -> str:
    """The brand's reply without the handle it opens with or the initials it signs off with."""
    tokens = text.split()
    while tokens and tokens[0] == "@USER":
        tokens.pop(0)
    return SIGNATURE.sub("", " ".join(tokens))


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def lcs_length(first: list[str], second: list[str]) -> int:
    """Length of the longest common subsequence, the L in ROUGE-L."""
    previous = [0] * (len(second) + 1)
    for token in first:
        current = [0]
        for index, other in enumerate(second):
            if token == other:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[-1], previous[index + 1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1 over lowercased word tokens."""
    left = words(reference)
    right = words(candidate)
    if not left or not right:
        return 0.0
    common = lcs_length(left, right)
    if common == 0:
        return 0.0
    precision = common / len(right)
    recall = common / len(left)
    return 2 * precision * recall / (precision + recall)


def section(runs: dict[str, dict[int, dict]], case_ids: list[int],
            cases: dict[int, dict]) -> tuple[list[str], dict]:
    """Mean ROUGE-L per system against the brand's real reply, with a bootstrap 95% interval."""
    systems = list(runs)
    lines = ["## 3.6 Reference similarity (diagnostic only, never a headline)", "",
             "ROUGE-L F1 between each system's reply and the brand's real reply, over "
             f"{len(case_ids)} cases. A constant DM hand-off would score well here without helping anyone, so "
             "this number ranks nothing.", "",
             "| system | ROUGE-L vs brand_reply [95% CI] |", "|---|---|"]
    results = {}
    for system in systems:
        scores = []
        for case_id in case_ids:
            reference = tidy_reference(cases[case_id]["brand_reply"])
            scores.append(rouge_l(reference, runs[system][case_id]["output"]["reply"]))
        vector = metrics.np.array(scores)
        low, high = metrics.bootstrap_ci(len(vector), metrics.mean_statistic(vector))
        lines.append(f"| {system} | {vector.mean():.3f} [{low:.3f}, {high:.3f}] |")
        results[system] = {"rouge_l": float(vector.mean()), "rouge_l_ci": [low, high]}
    return lines, results
