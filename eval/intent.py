"""Intent classification (REPORT.md 3.1), scored ONLY against the human labels: macro-F1 stratified and
reweighted, each with a bootstrap 95% CI, per-class F1 with support, and a confusion matrix per system."""

import numpy as np

from eval import data, metrics


def section(runs: dict[str, dict[int, dict]], case_ids: list[int], humans: dict[int, str],
            shares: dict[str, float], classes: list[str]) -> tuple[list[str], list, dict]:
    """Markdown lines, the paired-bootstrap comparisons, and the numbers for results.json."""
    scored = [case_id for case_id in case_ids if case_id in humans]
    truth = [humans[case_id] for case_id in scored]
    lines = ["## 3.1 Intent classification", "",
             f"Scored against {len(scored)} human labels only; {data.HUMAN_LABELS_PLANNED} are planned (the 40-case "
             "blind audit plus the 3 I labelled). The 147 model labels are never used here."]
    if len(scored) < data.HUMAN_LABELS_PLANNED:
        lines += ["", f"**The human labels are incomplete ({len(scored)} of {data.HUMAN_LABELS_PLANNED}): run "
                  "`python -m src.label audit`, then this again. Until then these figures mean little.**"]
    if not scored:
        return lines, [], {}
    weights = metrics.reweight(truth, shares)
    everything = np.arange(len(scored))
    lines += ["", "| system | macro-F1, stratified [95% CI] | macro-F1, reweighted [95% CI] |", "|---|---|---|"]
    predicted = {}
    statistics = {}
    results = {}
    for system, by_case in runs.items():
        predicted[system] = [by_case[case_id]["output"]["intent"] for case_id in scored]
        plain = metrics.macro_statistic(truth, predicted[system], classes)
        weighted = metrics.macro_statistic(truth, predicted[system], classes, weights)
        low, high = metrics.bootstrap_ci(len(scored), plain)
        weighted_low, weighted_high = metrics.bootstrap_ci(len(scored), weighted)
        lines.append(f"| {system} | {plain(everything):.3f} [{low:.3f}, {high:.3f}] | "
                     f"{weighted(everything):.3f} [{weighted_low:.3f}, {weighted_high:.3f}] |")
        statistics[system] = plain
        results[system] = {"n": len(scored), "macro_f1": plain(everything), "macro_f1_ci": [low, high],
                           "macro_f1_reweighted": weighted(everything),
                           "macro_f1_reweighted_ci": [weighted_low, weighted_high]}

    tables = {system: metrics.f1_by_class(truth, predicted[system], classes) for system in runs}
    lines += ["", "Per-class F1, with support in the human labels:", "",
              "| intent | support | " + " | ".join(runs) + " |", "|---" * (len(runs) + 2) + "|"]
    for intent in classes:
        support = int(tables["full"][intent]["support"])
        cells = [f"{tables[system][intent]['f1']:.3f}" if support else "-" for system in runs]
        lines.append(f"| {intent} | {support} | " + " | ".join(cells) + " |")

    for system in runs:
        matrix = metrics.confusion(truth, predicted[system], classes)
        lines += ["", f"Confusion matrix, {system}: rows are the human label, columns the prediction, numbered as "
                  "the rows.", "", "| human label | " + " | ".join(str(number) for number in range(1, len(classes) + 1))
                  + " |", "|---" * (len(classes) + 1) + "|"]
        for number, (intent, row) in enumerate(zip(classes, matrix), start=1):
            lines.append(f"| {number} {intent} | " + " | ".join(str(count) for count in row) + " |")
    return lines, [("macro-F1 on the human labels", len(scored), statistics)], results
