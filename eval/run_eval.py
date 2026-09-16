"""The evaluation (python -m eval.run_eval): REPORT.md Section 3 from the traces and judgements, with no model
call. Writes artifacts/eval/results.md and results.json, and prints the markdown.

Intent is scored ONLY against the human labels (eval/intent.py). Escalation and reply quality are scored on every
golden case that all four systems' runs cover.
"""

import json

import numpy as np

from eval import cost, data, escalation, intent, metrics, quality
from eval.judge import JUDGE_MODEL
from src import classify

RESULTS_DIR = data.TRACES_JSONL.parent / "eval"
PAIRS = [("full", "no_rag"), ("full", "b1"), ("full", "b0"), ("no_rag", "b1"), ("no_rag", "b0"), ("b1", "b0")]


def with_ci(vector: np.ndarray) -> str:
    low, high = metrics.bootstrap_ci(len(vector), metrics.mean_statistic(vector))
    return f"{np.nanmean(vector):.3f} [{low:.3f}, {high:.3f}]"


def escalation_section(runs: dict[str, dict[int, dict]], case_ids: list[int], gold: list[dict],
                       shares: dict[str, float]) -> tuple[list[str], list, dict]:
    """The table per system at its operating point, then each ladder system's grid and layer attribution."""
    n = len(case_ids)
    allowed = int(escalation.BUDGET_SHARE * n)
    weights = metrics.reweight([label["intent"] for label in gold], shares)
    relevant = sum(1 for label in gold if label["escalate"] and escalation.is_severe(label))
    lines = ["## 3.2 Escalation", "",
             f"{n} cases, {sum(1 for label in gold if label['escalate'])} labelled escalate, {relevant} of them SEVERE "
             f"if auto-handled. Pre-registered budget: 0 SEVERE and at most {allowed} ORDINARY harmful auto-replies. "
             "Rates are over all cases. B0 and B1 have no thresholds, so each is one point.", "",
             "| system | point (confidence, similarity) | within budget | auto-handled rate [95% CI] | SEVERE | "
             "ORDINARY | reweighted auto / SEVERE / ORDINARY | precision | recall |", "|---" * 9 + "|"]
    details = []
    statistics = {"auto-handle rate": {}, "SEVERE harmful rate": {}, "ORDINARY harmful rate": {}}
    results = {}
    for system, by_case in runs.items():
        traces = [by_case[case_id] for case_id in case_ids]
        if system in ("b0", "b1"):
            chosen = escalation.recorded_point(traces, gold)
            within = chosen.severe == 0 and chosen.ordinary <= allowed
        else:
            points = escalation.sweep(traces, gold)
            chosen = escalation.operating_point(points, n)
            within = chosen is not None
            if chosen is None:  # no grid point meets the budget: report the run's own thresholds instead
                chosen = escalation.default_point(points)
            details += [""] + escalation.grid_lines(system, points, n)
            details += [""] + escalation.attribution_lines(system, escalation.attribution(traces, gold, chosen))
        severe, ordinary = escalation.harms(chosen.escalate, gold)
        auto = [0.0 if flag else 1.0 for flag in chosen.escalate]
        precision, recall = escalation.precision_recall(chosen.escalate, gold)
        where = "-"
        if chosen.confidence is not None:
            where = f"{chosen.confidence:.2f}, " + ("-" if chosen.similarity is None else f"{chosen.similarity:.2f}")
        severe_cell = f"{chosen.severe} ({chosen.severe / n:.1%})"
        if chosen.severe == 0 and relevant:
            severe_cell += f"; rule-of-three 95% bound {3 / relevant:.1%}"
        reweighted = " / ".join(f"{metrics.weighted_mean(values, weights):.3f}" for values in (auto, severe, ordinary))
        lines.append(f"| {system} | {where} | {'yes' if within else 'no'} | {with_ci(np.array(auto))} | {severe_cell} | "
                     f"{chosen.ordinary} ({chosen.ordinary / n:.1%}) | {reweighted} | {precision:.3f} | {recall:.3f} |")
        statistics["auto-handle rate"][system] = metrics.mean_statistic(np.array(auto))
        statistics["SEVERE harmful rate"][system] = metrics.mean_statistic(np.array(severe, dtype=float))
        statistics["ORDINARY harmful rate"][system] = metrics.mean_statistic(np.array(ordinary, dtype=float))
        results[system] = {"confidence": chosen.confidence, "similarity": chosen.similarity, "within_budget": within,
                           "auto_handled": chosen.auto_handled, "severe": chosen.severe, "ordinary": chosen.ordinary,
                           "precision": precision, "recall": recall}
    comparisons = [(name, n, by_system) for name, by_system in statistics.items()]
    return lines + details, comparisons, results


def paired_section(comparisons: list) -> list[str]:
    lines = ["## Paired differences between systems", "",
             f"{metrics.RESAMPLES:,} paired bootstrap resamples (seed {metrics.SEED}): both systems are scored on the "
             "same resampled cases, each at its reported point. A difference stands only if its 95% interval "
             "excludes zero.", "", "| metric | a - b | difference | 95% CI | stands |", "|---|---|---|---|---|"]
    for name, n, statistics in comparisons:
        for first, second in PAIRS:
            if first in statistics and second in statistics:
                point, low, high = metrics.paired_difference(n, statistics[first], statistics[second])
                stands = "yes" if low > 0 or high < 0 else "no"
                lines.append(f"| {name} | {first} - {second} | {point:+.3f} | [{low:.3f}, {high:.3f}] | {stands} |")
    return lines


def main() -> None:
    labels = data.golden_labels()
    classes, _ = classify.parse_taxonomy(classify.TAXONOMY_MD.read_text(encoding="utf-8"))
    selected = data.latest_runs(data.SYSTEMS)
    covered = set.intersection(*[set(trace["case_id"] for trace in traces) for traces in selected.values()])
    case_ids = sorted(covered)
    runs = {}
    for system, traces in selected.items():
        runs[system] = {trace["case_id"]: trace for trace in traces if trace["case_id"] in covered}
    run_ids = {system: runs[system][case_ids[0]]["run_id"] for system in runs}
    judged = data.judgements(JUDGE_MODEL)
    judged_by_system = {system: [] for system in runs}
    for (system, run_id, _), record in judged.items():
        if system in runs and run_id == run_ids[system]:
            judged_by_system[system].append(record)
    shares = data.population_shares()
    gold = [labels[case_id] for case_id in case_ids]

    intent_lines, intent_pairs, intent_results = intent.section(runs, case_ids, data.human_intents(), shares, classes)
    escalation_lines, escalation_pairs, escalation_results = escalation_section(runs, case_ids, gold, shares)
    quality_lines, quality_pairs, quality_results = quality.section(runs, case_ids, labels, judged)
    cost_lines, cost_results = cost.section(runs, case_ids, judged_by_system, data.read_traces(),
                                            data.read_judgements())
    lines = ["# Evaluation results", "",
             "Written by `python -m eval.run_eval` from the traces and judgements, with no model call. Runs: "
             + ", ".join(f"{system} {run_id}" for system, run_id in run_ids.items())
             + f". {len(case_ids)} golden cases are covered by all four.", ""]
    lines += intent_lines + [""] + escalation_lines + [""] + quality_lines + [""]
    lines += paired_section(intent_pairs + escalation_pairs + quality_pairs) + [""] + cost_lines
    text = "\n".join(lines) + "\n"
    results = {"runs": run_ids, "cases": len(case_ids), "intent": intent_results, "escalation": escalation_results,
               "quality": quality_results, "cost": cost_results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.md").write_text(text, encoding="utf-8", newline="\n")
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
