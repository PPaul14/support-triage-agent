"""The full evaluation run, batched by model. scripts/run_all.ps1 starts it in the background, logging to
artifacts/run.log.

Order: B0 and B1, which call no model; the no-RAG ablation (phi3 classifies and gives L4, then llama3 drafts); the
full system (its phi3 prompts are the ablation's, so they come from the cache; then retrieval and llama3 drafts);
qwen2.5 judging every system's replies in one batch; then the metrics. Every LLM call goes through the disk cache,
so starting again after a stop resumes where the run stopped.
"""

import argparse
import statistics
import subprocess
import time
from datetime import datetime

from eval import cost, data, judge, run_eval
from src import baselines, pipeline
from src.runs import golden_cases

N_CASES = 150
PLAN = [  # (stage, the measured stage its median comes from, calls)
    ("no-RAG: classify (phi3)", "classify", N_CASES),
    ("no-RAG: L4 judgement (phi3), at most (L1 and L2 skip some)", "L4", N_CASES),
    ("no-RAG: drafts (llama3)", "no_rag draft", N_CASES),
    ("full: classify and L4 come from the cache", None, 0),
    ("full: drafts (llama3)", "full draft", N_CASES),
    ("judge (qwen2.5): B1, no-RAG and full, B0 once per intent", "judge", 3 * N_CASES + 9),
]


def log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def measured_latencies() -> dict[str, list[float]]:
    """Uncached call latencies measured so far (the smoke tests), per stage, with drafts kept apart per system."""
    samples: dict[str, list[float]] = {"classify": [], "L4": [], "no_rag draft": [], "full draft": [], "judge": []}
    for trace in data.read_traces():
        for stage, _, response in cost.llm_calls(trace):
            if not response["cache_hit"]:
                key = f"{trace['system']} draft" if stage == "draft" else stage
                samples[key].append(response["latency_s"])
    for record in data.read_judgements():
        if record["response"] is not None and not record["response"]["cache_hit"]:
            samples["judge"].append(record["response"]["latency_s"])
    return samples


def projection() -> None:
    samples = measured_latencies()
    print("Projected wall clock for the full run: the median of the uncached calls measured so far, times the calls.")
    print("A projection, not a measurement: model loads, draft retries and calls already cached are left out.\n")
    print(f"{'stage':62} {'calls':>6} {'median s':>9} {'from n':>7} {'hours':>6}")
    total = 0.0
    for label, stage, calls in PLAN:
        if stage is None:
            print(f"{label:62} {calls:>6}")
            continue
        if not samples[stage]:
            print(f"{label:62} {calls:>6} {'not measured yet':>24}")
            continue
        median = statistics.median(samples[stage])
        hours = calls * median / 3600
        total += hours
        print(f"{label:62} {calls:>6} {median:>9.1f} {len(samples[stage]):>7} {hours:>6.2f}")
    print(f"{'total':62} {'':>6} {'':>9} {'':>7} {total:>6.2f}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="The full evaluation run, batched by model.")
    parser.add_argument("--projection-only", action="store_true", help="print the projected wall clock and stop")
    args = parser.parse_args()
    projection()
    if args.projection_only:
        return
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                             text=True).stdout.strip()
    log(f"run started at commit {commit}" + (", WITH UNCOMMITTED CHANGES" if changed else ""))
    cases = golden_cases()
    stages = [("B0 and B1", lambda: baselines.run_baselines(cases)),
              ("no-RAG ablation", lambda: pipeline.run_batch(cases, no_rag=True)),
              ("full system", lambda: pipeline.run_batch(cases)),
              ("qwen2.5 judge", lambda: judge.judge_runs(data.SYSTEMS)),
              ("metrics", run_eval.main)]
    for name, work in stages:
        start = time.perf_counter()
        log(f"start: {name}")
        work()
        log(f"done: {name}, {(time.perf_counter() - start) / 60:.1f} min")
    log("run finished: artifacts/eval/results.md")


if __name__ == "__main__":
    main()
