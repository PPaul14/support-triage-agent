"""Reply quality (REPORT.md 3.3) from the judge's verdicts: per system, the rate of "yes" on each check, with a
bootstrap 95% CI on would_send_unedited, the headline quality metric. B0's verdict for every case is the one given
on its golden intent's representative case (eval/data.py)."""

import math

import numpy as np

from eval import data, metrics
from eval.judge import CHECKS, JUDGE_MODEL


def answers_per_case(system: str, by_case: dict[int, dict], case_ids: list[int], labels: dict[int, dict],
                     judged: dict[tuple[str, str, int], dict]) -> list[dict | None]:
    """Per case, the judge's answers; None when the reply was not judged or the judgement was unusable."""
    run_id = by_case[case_ids[0]]["run_id"]
    representatives = data.b0_representatives(case_ids, labels)
    found = []
    for case_id in case_ids:
        judged_id = representatives[labels[case_id]["intent"]] if system == "b0" else case_id
        record = judged.get((system, run_id, judged_id))
        found.append(record["answers"] if record is not None and record["usable"] else None)
    return found


def section(runs: dict[str, dict[int, dict]], case_ids: list[int], labels: dict[int, dict],
            judged: dict[tuple[str, str, int], dict]) -> tuple[list[str], list, dict]:
    """Markdown lines, the paired-bootstrap comparisons, and the numbers for results.json."""
    systems = list(runs)
    lines = ["## 3.3 Reply quality", "",
             f"Judge: {JUDGE_MODEL}, one call per reply, five binary checks. Each cell is the rate of \"yes\" over "
             "the usable judgements; a good reply answers no to contains_unsupported_specific and yes to the rest.",
             "", "| check | " + " | ".join(systems) + " |", "|---" * (len(systems) + 1) + "|"]
    vectors: dict[str, dict[str, np.ndarray]] = {}
    usable = {}
    for system in systems:
        answers = answers_per_case(system, runs[system], case_ids, labels, judged)
        usable[system] = sum(1 for found in answers if found is not None)
        vectors[system] = {}
        for check in CHECKS:
            values = [math.nan if found is None else float(found[check] == "yes") for found in answers]
            vectors[system][check] = np.array(values)
    results: dict[str, dict] = {system: {"usable": usable[system]} for system in systems}
    for check in CHECKS:
        cells = []
        for system in systems:
            vector = vectors[system][check]
            if usable[system] == 0:
                cells.append("not judged")
                continue
            rate = float(np.nanmean(vector))
            results[system][check] = rate
            if check == "would_send_unedited":
                low, high = metrics.bootstrap_ci(len(vector), metrics.mean_statistic(vector))
                results[system]["would_send_unedited_ci"] = [low, high]
                cells.append(f"{rate:.3f} [{low:.3f}, {high:.3f}]")
            else:
                cells.append(f"{rate:.3f}")
        lines.append(f"| {check} | " + " | ".join(cells) + " |")
    lines.append("| usable judgements | " + " | ".join(f"{usable[system]} of {len(case_ids)}" for system in systems)
                 + " |")
    lines += ["", "B0 was judged once per golden intent, and each verdict is reused for every case of that intent."]
    statistics = {}
    for system in systems:
        if usable[system]:
            statistics[system] = metrics.mean_statistic(vectors[system]["would_send_unedited"])
    return lines, [("would_send_unedited rate", len(case_ids), statistics)], results
