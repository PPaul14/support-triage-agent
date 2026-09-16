"""Reply quality: qwen2.5:7b answers five binary checks per reply (python -m eval.judge; REPORT.md 3.3).

Evidence is the case (the customer's message and its two latest earlier turns) plus the precedents the system was
given. The no-RAG ablation and the baselines were given none, so theirs is the case alone: B1's copied reply came
from a neighbour it never showed as evidence. The judge quotes evidence only for an answer that fails the reply.
B0's constant reply is judged once per golden intent and the verdict reused. One batch judges every system, so the
judge model loads once. Verdicts are appended to artifacts/judgements.jsonl.
"""

import argparse
import hashlib
import time
from dataclasses import dataclass

from eval.data import JUDGEMENTS_JSONL, SYSTEMS, b0_representatives, golden_labels, latest_runs
from src import llm
from src.runs import golden_cases, progress, write_trace

JUDGE_MODEL = "qwen2.5:7b-instruct-q4_K_M"
NUM_PREDICT = 200  # raised from 150: with the quote mandatory, judgements reached 142 tokens
CHECKS = {  # name: (question, the answer that fails the reply), worded as in REPORT.md 3.3
    "contains_unsupported_specific": ("Does the reply state a number, date, duration or policy that is absent "
                                      "from the evidence?", "yes"),
    "advances_resolution": ("Does the reply give the customer a concrete step or answer to act on? A reply "
                            "that only moves the conversation to DM does not.", "no"),
    "addresses_stated_problem": ("Does the reply respond to the problem the customer actually raised?", "no"),
    "tone_appropriate": ("Is the reply polite and in the brand's voice?", "no"),
    "would_send_unedited": ("Could the reply be sent as written, with no edits?", "no"),
}


@dataclass
class Judgement:
    run_id: str  # the system run whose reply was judged
    system: str
    judged_case_id: int  # for B0, the one case per golden intent its constant reply was judged on
    model: str
    usable: bool  # False when the judge's JSON was still invalid after one repair
    answers: dict[str, str]  # check -> "yes" or "no"
    evidence: dict[str, str]
    missing_evidence: list[str]  # checks answered the failing way without the required quote
    prompt_sha256: str
    response: llm.LLMResponse | None


def rubric() -> str:
    """The static start of every judge prompt, so Ollama can reuse it between calls."""
    lines = ["You grade a reply drafted for Spotify's customer support on Twitter. Judge only from the EVIDENCE "
             "and the REPLY below, and answer every check with yes or no.", ""]
    for name, (question, _) in CHECKS.items():
        lines.append(f"{name}: {question}")
    lines += ["", "An answer fails the reply when it is yes on contains_unsupported_specific, or no on any "
              "other check. Every failing answer MUST carry, in its own \"evidence\" field, a quote of under 12 "
              "words from the reply or the evidence that decides it. Leave \"evidence\" empty only where the "
              "answer does not fail. Reply with JSON only.", "", ""]
    return "\n".join(lines)


def schema() -> dict:
    check = {"type": "object", "required": ["answer", "evidence"],
             "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}, "evidence": {"type": "string"}}}
    return {"type": "object", "properties": {name: check for name in CHECKS}, "required": list(CHECKS)}


def judge_prompt(case: dict, hits: list[dict], reply: str) -> str:
    """The rubric first, then the evidence, then the reply to grade LAST."""
    lines = ["=== EVIDENCE: THIS CONVERSATION ==="]
    for turn in case["prior_turns"][-2:]:
        lines.append(f"Earlier {turn['role']} message: {turn['text']}")
    lines.append(f"Customer message: {case['customer_text']}")
    if hits:
        lines += ["", "=== EVIDENCE: OTHER CUSTOMERS' PAST CASES THE AGENT WAS GIVEN ==="]
        for number, hit in enumerate(hits, start=1):
            lines.append(f"[{number}] Another customer wrote: {hit['precedent']['customer_text']}")
            lines.append(f"[{number}] The brand replied: {hit['precedent']['brand_reply']}")
    lines += ["", "=== REPLY TO GRADE ===", reply]
    return rubric() + "\n".join(lines)


def judge_reply(run_id: str, system: str, case: dict, hits: list[dict], reply: str, model: str) -> Judgement:
    prompt = judge_prompt(case, hits, reply)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    try:
        response = llm.complete(prompt, model, schema=schema(), num_predict=NUM_PREDICT)
    except llm.LLMParseError:
        return Judgement(run_id, system, case["case_id"], model, False, {}, {}, [], digest, None)
    answers = {name: response.parsed[name]["answer"] for name in CHECKS}
    evidence = {name: response.parsed[name]["evidence"].strip() for name in CHECKS}
    missing = [name for name, (_, failing) in CHECKS.items() if answers[name] == failing and not evidence[name]]
    return Judgement(run_id, system, case["case_id"], model, True, answers, evidence, missing, digest, response)


def jobs(runs: dict[str, list[dict]], labels: dict[int, dict]) -> list[tuple[str, dict]]:
    """(system, trace) pairs to judge, in SYSTEMS order; B0 only on its one case per golden intent."""
    work = []
    for system, traces in runs.items():
        if system == "b0":
            keep = set(b0_representatives([trace["case_id"] for trace in traces], labels).values())
            traces = [trace for trace in traces if trace["case_id"] in keep]
        for trace in traces:
            work.append((system, trace))
    return work


def judge_runs(systems: list[str], model: str = JUDGE_MODEL) -> None:
    """Judge every reply of each system's largest run in one batch, appending to JUDGEMENTS_JSONL."""
    labels = golden_labels()
    cases = {case["case_id"]: case for case in golden_cases()}
    work = jobs(latest_runs(systems), labels)
    start = time.perf_counter()
    unusable = 0
    missing = 0
    for number, (system, trace) in enumerate(work, start=1):
        retrieval = trace.get("retrieval")  # baseline traces have none
        hits = [] if retrieval is None else retrieval["hits"]
        judgement = judge_reply(trace["run_id"], system, cases[trace["case_id"]], hits, trace["output"]["reply"],
                                model)
        write_trace(judgement, JUDGEMENTS_JSONL)
        if not judgement.usable:
            unusable += 1
        missing += len(judgement.missing_evidence)
        progress(f"judge {system}", number, len(work), start)
    print(f"judged {len(work)} replies with {model}: {unusable} unusable, {missing} failing answers without the "
          "required quote", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge each system's replies from its largest run.")
    parser.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    parser.add_argument("--model", default=JUDGE_MODEL)
    args = parser.parse_args()
    judge_runs(args.systems, args.model)


if __name__ == "__main__":
    main()
