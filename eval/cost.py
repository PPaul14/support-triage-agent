"""Cost and throughput (REPORT.md 3.5), from uncached LLM calls only: a cache hit's latency is a disk read.

Each call a system makes is matched, by stage and prompt hash, to the uncached call that produced it, in whichever
run that happened: the full system reuses the ablation's phi3 calls from the cache, but running alone it would pay
for them. Tickets per hour counts the stages a live system runs per ticket; judging is evaluation only.
"""

import math
import statistics
from dataclasses import asdict, dataclass

STAGES = ["classify", "L4", "draft", "judge"]
LIVE_STAGES = ["classify", "L4", "draft"]


@dataclass
class StageCost:
    calls: int
    measured: int  # calls matched to an uncached measurement
    seconds: float  # summed latency of the matched measurements
    median_s: float
    prompt_tokens: int
    completion_tokens: int


def llm_calls(trace: dict) -> list[tuple[str, str, dict]]:
    """(stage, prompt_sha256, response) for every LLM call in one trace. Baseline traces have none."""
    if "prediction" not in trace:
        return []
    calls = [("classify", trace["prediction"]["prompt_sha256"], trace["prediction"]["response"])]
    if trace["judgement"] is not None:
        calls.append(("L4", trace["judgement"]["prompt_sha256"], trace["judgement"]["response"]))
    for attempt in trace["draft"]["attempts"]:
        calls.append(("draft", attempt["prompt_sha256"], attempt["response"]))
    return calls


def measurements(all_traces: list[dict], all_judgements: list[dict]) -> dict[tuple[str, str], dict]:
    """(stage, prompt hash) -> the uncached response that measured it, from every run so far."""
    found = {}
    for trace in all_traces:
        for stage, digest, response in llm_calls(trace):
            if not response["cache_hit"]:
                found[(stage, digest)] = response
    for record in all_judgements:
        response = record["response"]
        if response is not None and not response["cache_hit"]:
            found[("judge", record["prompt_sha256"])] = response
    return found


def summarise(matched: list[dict | None]) -> StageCost:
    measured = [response for response in matched if response is not None]
    latencies = [response["latency_s"] for response in measured]
    return StageCost(len(matched), len(measured), sum(latencies),
                     statistics.median(latencies) if latencies else math.nan,
                     sum(response["prompt_tokens"] for response in measured),
                     sum(response["completion_tokens"] for response in measured))


def seconds_per_ticket(by_case: dict[int, dict], case_ids: list[int], costs: dict[str, StageCost]) -> float:
    """Mean seconds per ticket over the live stages: measured LLM time, scaled for calls without a measurement,
    plus the per-case retrieval search. The baselines' own recorded time for them."""
    if "prediction" not in by_case[case_ids[0]]:
        return sum(by_case[case_id]["latency_s"] for case_id in case_ids) / len(case_ids)
    total = 0.0
    for stage in LIVE_STAGES:
        stage_cost = costs[stage]
        if stage_cost.calls and not stage_cost.measured:
            return math.nan
        if stage_cost.calls:
            total += stage_cost.seconds / stage_cost.measured * stage_cost.calls
    total += sum(by_case[case_id]["latency_s"]["retrieve"] for case_id in case_ids)
    return total / len(case_ids)


def section(runs: dict[str, dict[int, dict]], case_ids: list[int], judged_by_system: dict[str, list[dict]],
            all_traces: list[dict], all_judgements: list[dict]) -> tuple[list[str], dict]:
    found = measurements(all_traces, all_judgements)
    lines = ["## 3.5 Cost and throughput", "",
             "From uncached calls only: each call a system makes is matched, by stage and prompt hash, to the uncached "
             "call that produced it (eval/cost.py). Tickets per hour count classify, L4 where asked, retrieval and "
             "the draft; judging is left out. Hosted-model cost: TBD (it needs a named model and a dated price).", "",
             "| system | stage | calls | measured | median s/call | total s | prompt tokens | output tokens |",
             "|---" * 8 + "|"]
    results = {}
    for system, by_case in runs.items():
        matched: dict[str, list[dict | None]] = {stage: [] for stage in STAGES}
        for case_id in case_ids:
            for stage, digest, _ in llm_calls(by_case[case_id]):
                matched[stage].append(found.get((stage, digest)))
        for record in judged_by_system[system]:
            matched["judge"].append(found.get(("judge", record["prompt_sha256"])))
        costs = {stage: summarise(records) for stage, records in matched.items()}
        for stage, stage_cost in costs.items():
            if stage_cost.calls:
                lines.append(f"| {system} | {stage} | {stage_cost.calls} | {stage_cost.measured} | "
                             f"{stage_cost.median_s:.1f} | {stage_cost.seconds:.0f} | {stage_cost.prompt_tokens:,} | "
                             f"{stage_cost.completion_tokens:,} |")
        per_ticket = seconds_per_ticket(by_case, case_ids, costs)
        results[system] = {"stages": {stage: asdict(stage_cost) for stage, stage_cost in costs.items()},
                           "seconds_per_ticket": per_ticket,
                           "tickets_per_hour": 3600 / per_ticket if per_ticket > 0 else math.nan}
    lines += ["", "| system | seconds per ticket | tickets per hour, this machine |", "|---|---|---|"]
    for system, row in results.items():
        lines.append(f"| {system} | {row['seconds_per_ticket']:.3f} | {row['tickets_per_hour']:,.0f} |")
    return lines, results
