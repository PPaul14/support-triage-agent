"""The agent end to end: classify, escalate, retrieve and draft, with one trace record per case.

python -m src.pipeline --limit 5 [--no-rag]   (scripts/smoke.ps1 runs the smoke test offline)

Stages run as batches, one model at a time, never alternating per case: phi3 classifies every case and gives
every L4 judgement, then llama3 drafts every reply. --no-rag is the ablation in REPORT.md Section 3: it
retrieves nothing, so its drafts get no precedents and its ladder has no similarity gate.
"""

import argparse
import time
from dataclasses import dataclass

from src import classify, escalate, generate, index, retrieve
from src.runs import TRACES_JSONL, golden_cases, new_run_id, progress, write_trace


@dataclass
class AgentOutput:
    case_id: int
    system: str  # "full", "no_rag", "b0" or "b1"
    intent: str
    escalate: bool
    reason_code: str
    reason_text: str
    layer: int | None  # the escalation layer that decided; None for the baselines
    reply: str  # drafted for every case: sent when auto-handled, a suggestion for the human otherwise
    precedent_id: int | None


@dataclass
class Trace:
    run_id: str
    system: str
    case_id: int
    output: AgentOutput
    prediction: classify.Prediction
    retrieval: retrieve.Retrieval | None  # None in the no-RAG ablation
    l1: escalate.Decision | None  # each layer's own verdict, so any layer can be re-scored offline
    l2: escalate.Decision | None
    l3: escalate.Decision | None
    judgement: escalate.Judgement | None  # None when L1 or L2 fired: phi3 was not asked
    draft: generate.Draft
    latency_s: dict[str, float]  # this case's time per stage; an LLM cache hit's figure is a disk read


def run(case: dict) -> AgentOutput:
    """One case. The two models alternate here, so use run_batch for more than one case."""
    return run_batch([case])[0]


def run_batch(cases: list[dict], no_rag: bool = False) -> list[AgentOutput]:
    """Every case through each stage in turn; one trace per case is appended to TRACES_JSONL."""
    system = "no_rag" if no_rag else "full"
    run_id = new_run_id()
    taxonomy_text = classify.TAXONOMY_MD.read_text(encoding="utf-8")
    intents, _ = classify.parse_taxonomy(taxonomy_text)
    block = classify.render_block(taxonomy_text)
    stage_s: dict[str, float] = {}

    start = time.perf_counter()  # phi3: an intent for every case
    predictions = []
    for number, case in enumerate(cases, start=1):
        predictions.append(classify.predict(case, block, intents))
        progress(f"{system} classify", number, len(cases), start)
    stage_s["classify"] = time.perf_counter() - start

    start = time.perf_counter()  # phi3 still loaded. L1 and L2 need no model; L4 is asked where neither fired
    firsts = [escalate.layer1(case) for case in cases]
    seconds = [escalate.layer2(case, prediction.intent) for case, prediction in zip(cases, predictions)]
    judgements: list[escalate.Judgement | None] = []
    for number, (case, prediction, first, second) in enumerate(zip(cases, predictions, firsts, seconds), start=1):
        judgements.append(escalate.judge(case, prediction.intent) if first is None and second is None else None)
        progress(f"{system} L4", number, len(cases), start)
    stage_s["judge"] = time.perf_counter() - start

    start = time.perf_counter()  # CPU: MiniLM embeddings and a flat cosine search
    retrievals: list[retrieve.Retrieval | None] = [None] * len(cases)
    if not no_rag:
        built = index.build()
        vectors = retrieve.embed_cases(cases)
        retrievals = [retrieve.retrieve(built, vector, p.intent) for vector, p in zip(vectors, predictions)]
    stage_s["retrieve"] = time.perf_counter() - start

    start = time.perf_counter()  # llama3: a draft for every case, escalated or not, so every reply can be judged
    drafts = []
    for number, (case, prediction, retrieval) in enumerate(zip(cases, predictions, retrievals), start=1):
        drafts.append(generate.draft(case, prediction.intent, [] if retrieval is None else retrieval.hits))
        progress(f"{system} draft", number, len(cases), start)
    stage_s["draft"] = time.perf_counter() - start

    outputs = []
    for case, prediction, first, second, judgement, retrieval, draft in zip(
            cases, predictions, firsts, seconds, judgements, retrievals, drafts):
        max_sim = None if retrieval is None else retrieval.max_sim
        third = escalate.layer3(prediction.confidence, max_sim, retrieve.SIMILARITY_FLOOR, draft.accepted)
        final = escalate.decide(first, second, third, None if judgement is None else judgement.decision)
        output = AgentOutput(case["case_id"], system, prediction.intent, final.escalate, final.reason_code,
                             final.reason_text, final.layer, draft.text, draft.precedent_id)
        latency = {"classify": prediction.response.latency_s,
                   "judge": 0.0 if judgement is None else judgement.response.latency_s,
                   "retrieve": 0.0 if retrieval is None else retrieval.latency_s,
                   "draft": sum(attempt.response.latency_s for attempt in draft.attempts)}
        write_trace(Trace(run_id, system, case["case_id"], output, prediction, retrieval, first, second, third,
                          judgement, draft, latency))
        outputs.append(output)

    print(f"\n{system} run {run_id}: {len(cases)} cases, traces appended to {TRACES_JSONL}")
    for stage, seconds_taken in stage_s.items():
        print(f"  {stage:9} {seconds_taken:8.1f} s wall clock, {seconds_taken / len(cases):6.1f} s per case")
    total = sum(stage_s.values())
    print(f"  {'total':9} {total:8.1f} s wall clock, {total / len(cases):6.1f} s per case", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent on golden cases and append one trace per case.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N golden cases (default: all 150)")
    parser.add_argument("--no-rag", action="store_true", help="the ablation: no retrieval and no precedents")
    args = parser.parse_args()
    run_batch(golden_cases(args.limit), no_rag=args.no_rag)


if __name__ == "__main__":
    main()
