"""Escalation, re-scored offline from the traces (REPORT.md 3.2).

The ladder is re-applied at any pair of thresholds from each trace's recorded layer verdicts, so the operating
curve needs no model call. At the thresholds the run itself used it must reproduce every recorded decision."""

import math
from dataclasses import dataclass

from src.escalate import CONFIDENCE_THRESHOLD
from src.retrieve import SIMILARITY_FLOOR

CONFIDENCE_GRID = [round(0.50 + 0.05 * step, 2) for step in range(10)]  # 0.50 to 0.95
SIMILARITY_GRID = [round(SIMILARITY_FLOOR + 0.05 * step, 2) for step in range(9)]  # the floor, 0.50, to 0.90
BUDGET_SHARE = 0.05  # ORDINARY harmful auto-replies allowed, as a share of all cases; SEVERE allowed: none


@dataclass
class Point:
    confidence: float | None  # None: no confidence threshold (B0, B1)
    similarity: float | None  # None: no similarity gate (B0, B1, the no-RAG ablation)
    escalate: list[bool]  # per case, in case_id order
    auto_handled: int
    severe: int
    ordinary: int


def is_severe(label: dict) -> bool:
    return label["intent"] == "billing_subscription" or label["compromised"]


def layer_verdicts(trace: dict, confidence: float, similarity: float | None) -> dict[int, bool | None]:
    """Each layer's own verdict at these thresholds. L4 is None where phi3 was not asked (L1 or L2 fired)."""
    retrieval = trace["retrieval"]
    below = similarity is not None and retrieval is not None and retrieval["max_sim"] < similarity
    third = trace["prediction"]["confidence"] < confidence or below or not trace["draft"]["accepted"]
    judgement = trace["judgement"]
    return {1: trace["l1"] is not None, 2: trace["l2"] is not None, 3: third,
            4: None if judgement is None else judgement["decision"]["escalate"]}


def decide(trace: dict, confidence: float, similarity: float | None) -> bool:
    """The ladder: the first layer that fires decides, and L4 decides only where L1 to L3 all pass."""
    verdicts = layer_verdicts(trace, confidence, similarity)
    if verdicts[1] or verdicts[2] or verdicts[3]:
        return True
    if verdicts[4] is None:
        raise RuntimeError(f"case {trace['case_id']}: L1 to L3 pass, but phi3 was never asked for L4")
    return verdicts[4]


def harms(escalate: list[bool], labels: list[dict]) -> tuple[list[int], list[int]]:
    """Per case, 1 for a SEVERE and 1 for an ORDINARY harmful auto-reply: labelled escalate, but auto-handled."""
    severe = []
    ordinary = []
    for escalated, label in zip(escalate, labels):
        harmful = label["escalate"] and not escalated
        severe.append(int(harmful and is_severe(label)))
        ordinary.append(int(harmful and not is_severe(label)))
    return severe, ordinary


def make_point(escalate: list[bool], labels: list[dict], confidence: float | None = None,
               similarity: float | None = None) -> Point:
    severe, ordinary = harms(escalate, labels)
    return Point(confidence, similarity, escalate, escalate.count(False), sum(severe), sum(ordinary))


def recorded_point(traces: list[dict], labels: list[dict]) -> Point:
    """B0 and B1: the decisions as run, a single point."""
    return make_point([trace["output"]["escalate"] for trace in traces], labels)


def sweep(traces: list[dict], labels: list[dict]) -> list[Point]:
    """Every grid point for a ladder system, once the run's own thresholds have reproduced every decision."""
    has_retrieval = traces[0]["retrieval"] is not None
    run_similarity = SIMILARITY_FLOOR if has_retrieval else None
    wrong = [trace["case_id"] for trace in traces
             if decide(trace, CONFIDENCE_THRESHOLD, run_similarity) != trace["output"]["escalate"]]
    if wrong:
        raise RuntimeError(f"offline re-scoring disagrees with the recorded decisions on cases {wrong}")
    similarities = SIMILARITY_GRID if has_retrieval else [None]
    points = []
    for confidence in CONFIDENCE_GRID:
        for similarity in similarities:
            escalate = [decide(trace, confidence, similarity) for trace in traces]
            points.append(make_point(escalate, labels, confidence, similarity))
    return points


def operating_point(points: list[Point], n: int) -> Point | None:
    """The grid point with the most auto-handled cases within the budget: no SEVERE and at most floor(5% of n)
    ORDINARY harmful auto-replies. Ties go to fewer ORDINARY, then to the lower thresholds. None if none fits."""
    allowed = int(BUDGET_SHARE * n)  # 5% of 150 is 7.5, so at most 7
    within = [point for point in points if point.severe == 0 and point.ordinary <= allowed]
    if not within:
        return None
    return min(within, key=lambda point: (-point.auto_handled, point.ordinary, point.confidence,
                                          point.similarity or 0.0))


def default_point(points: list[Point]) -> Point:
    """The grid point at the thresholds the run itself used."""
    for point in points:
        if point.confidence == CONFIDENCE_THRESHOLD and point.similarity in (SIMILARITY_FLOOR, None):
            return point
    raise RuntimeError("the run's own thresholds are not on the grid")


def precision_recall(escalate: list[bool], labels: list[dict]) -> tuple[float, float]:
    """Escalate is the positive class."""
    caught = sum(1 for flag, label in zip(escalate, labels) if flag and label["escalate"])
    flagged = sum(1 for flag in escalate if flag)
    actual = sum(1 for label in labels if label["escalate"])
    return (caught / flagged if flagged else math.nan), (caught / actual if actual else math.nan)


def attribution(traces: list[dict], labels: list[dict], point: Point) -> dict[int, dict[str, int]]:
    """Per layer, at this point: the cases it fired on, and the escalate-labelled cases it alone caught."""
    table = {layer: {"fired": 0, "alone_caught": 0} for layer in (1, 2, 3, 4)}
    for trace, label in zip(traces, labels):
        verdicts = layer_verdicts(trace, point.confidence, point.similarity)
        firing = [layer for layer, verdict in verdicts.items() if verdict]
        for layer in firing:
            table[layer]["fired"] += 1
        if label["escalate"] and len(firing) == 1:
            table[firing[0]]["alone_caught"] += 1
    return table


def grid_lines(system: str, points: list[Point], n: int) -> list[str]:
    """The operating curve as a table: auto-handled / SEVERE / ORDINARY at every grid point."""
    cell = {(point.confidence, point.similarity): point for point in points}
    similarities = SIMILARITY_GRID if points[0].similarity is not None else [None]
    heads = ["no similarity gate" if value is None else f"similarity {value:.2f}" for value in similarities]
    lines = [f"Operating curve, {system}: auto-handled / SEVERE / ORDINARY, of {n} cases.", "",
             "| confidence | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
    for confidence in CONFIDENCE_GRID:
        values = [cell[(confidence, value)] for value in similarities]
        lines.append(f"| {confidence:.2f} | " + " | ".join(f"{p.auto_handled}/{p.severe}/{p.ordinary}" for p in values)
                     + " |")
    return lines


def attribution_lines(system: str, table: dict[int, dict[str, int]]) -> list[str]:
    lines = [f"Layer attribution, {system}, at the point reported above:", "",
             "| layer | fired | alone caught (labelled escalate) |", "|---|---|---|"]
    for layer, row in table.items():
        lines.append(f"| L{layer} | {row['fired']} | {row['alone_caught']} |")
    return lines + ["", "L4 was not asked where L1 or L2 fired, so \"alone\" there means alone among those consulted."]
