"""Evaluation inputs, read-only: golden labels, the human intent labels, traces, judgements, population shares."""

import random
from collections import Counter

from src import classify, golden
from src.label import AUDIT_JSONL, GOLDEN_JSONL, read_jsonl
from src.runs import TRACES_JSONL

JUDGEMENTS_JSONL = TRACES_JSONL.parent / "judgements.jsonl"  # quotes tweets as evidence: gitignored
SYSTEMS = ["b0", "b1", "no_rag", "full"]
HUMAN_LABELS_PLANNED = 43  # the 40-case blind audit plus the 3 cases I labelled


def golden_labels() -> dict[int, dict]:
    return {record["case_id"]: record for record in read_jsonl(GOLDEN_JSONL)}


def human_intents() -> dict[int, str]:
    """The only intent ground truth: my blind audit labels, plus the golden rows I labelled myself ("manual" or
    "assisted"; the two earliest rows predate the mode field and are manual). Model labels are never included."""
    intents = {}
    for record in read_jsonl(GOLDEN_JSONL):
        if record.get("mode", "manual") in ("manual", "assisted"):
            intents[record["case_id"]] = record["intent"]
    for record in read_jsonl(AUDIT_JSONL):
        intents[record["case_id"]] = record["audit_intent"]
    return intents


def read_traces() -> list[dict]:
    return read_jsonl(TRACES_JSONL)


def latest_runs(systems: list[str]) -> dict[str, list[dict]]:
    """Per system, the traces of its largest run (the most cases; the latest on a tie), sorted by case_id."""
    runs: dict[tuple[str, str], list[dict]] = {}
    for trace in read_traces():
        runs.setdefault((trace["system"], trace["run_id"]), []).append(trace)
    chosen = {}
    for system in systems:
        candidates = [(len(traces), run_id) for (name, run_id), traces in runs.items() if name == system]
        if not candidates:
            raise SystemExit(f"no traces for system {system!r} in {TRACES_JSONL}")
        _, run_id = max(candidates)
        chosen[system] = sorted(runs[(system, run_id)], key=lambda trace: trace["case_id"])
    return chosen


def read_judgements() -> list[dict]:
    """Every judgement record ever written, repeats included: cost needs the first, uncached one."""
    return read_jsonl(JUDGEMENTS_JSONL)


def judgements(model: str) -> dict[tuple[str, str, int], dict]:
    """(system, run_id, judged_case_id) -> the latest judgement by this judge model."""
    found = {}
    for record in read_judgements():
        if record["model"] == model:
            found[(record["system"], record["run_id"], record["judged_case_id"])] = record
    return found


def population_shares() -> dict[str, float]:
    """phi3's estimated intent shares over the sampler's 1,500-case pool: the reweighting target (REPORT 3.1).
    Read from the disk cache: no model call."""
    taxonomy_text = classify.TAXONOMY_MD.read_text(encoding="utf-8")
    order, _ = classify.parse_taxonomy(taxonomy_text)
    eligible, _, _, _ = golden.load_eligible(taxonomy_text)
    pool = random.Random(golden.SEED).sample(eligible, golden.ESTIMATE_POOL_SIZE)  # the sampler's first draw
    estimates = golden.cached_estimates(pool, classify.render_block(taxonomy_text), order)
    counts = Counter(estimates.values())
    return {intent: counts[intent] / len(estimates) for intent in order}


def b0_representatives(case_ids: list[int], labels: dict[int, dict]) -> dict[str, int]:
    """Golden intent -> its smallest case_id: the one case per intent that B0's constant reply is judged on."""
    chosen: dict[str, int] = {}
    for case_id in sorted(case_ids):
        chosen.setdefault(labels[case_id]["intent"], case_id)
    return chosen
