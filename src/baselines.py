"""Baselines with zero LLM calls (python -m src.baselines --limit 5).

B0, trivial: the majority intent, one constant reply, never escalates.
B1, simple: TF-IDF + logistic regression for intent, a copy of the nearest past case's brand reply, and
escalation on a keyword list.
Both train on phi3's cached estimates and copy only from the leak-safe cases the precedent index holds.
"""

import argparse
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import index
from src.pipeline import AgentOutput
from src.runs import golden_cases, new_run_id, write_trace

B0_REPLY = ("Hi there, sorry for the trouble! Could you tell us a bit more about what's happening, and which "
            "device and app version you're using? We'll take it from there.")
B1_KEYWORDS = ["refund", "charge", "charged", "charges", "billing", "billed", "payment", "paid", "subscription",
               "cancel", "family plan", "student", "hacked", "hacker", "fraud", "scam", "lawyer", "gdpr"]
B1_ESCALATE = re.compile(r"\b(?:" + "|".join(re.escape(keyword) for keyword in B1_KEYWORDS) + r")\b", re.IGNORECASE)


@dataclass
class Fitted:
    majority_intent: str
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    cases: list[dict]  # the leak-safe cases, sorted by case_id: the only replies B1 may copy
    matrix: object  # their TF-IDF rows, a scipy sparse matrix with unit-length rows


@dataclass
class BaselineTrace:
    run_id: str
    system: str
    case_id: int
    output: AgentOutput
    neighbour_similarity: float | None  # B1 only: TF-IDF cosine to the case whose reply was copied
    latency_s: float


def fit() -> Fitted:
    cases, _ = index.precedent_cases()
    labels = index.silver_labels()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform([index.embedding_text(case["customer_text"]) for case in cases])
    row_of = {case["case_id"]: row for row, case in enumerate(cases)}
    train_ids = sorted(case_id for case_id in labels if case_id in row_of)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(matrix[[row_of[case_id] for case_id in train_ids]], [labels[case_id] for case_id in train_ids])
    counts = Counter(labels.values())
    majority = sorted(counts, key=lambda intent: (-counts[intent], intent))[0]  # ties broken by name
    return Fitted(majority, vectorizer, classifier, cases, matrix)


def without_leading_handles(text: str) -> str:
    """A past reply opens with the customer's @USER; the drafted replies have none, so neither does B1's copy."""
    tokens = text.split()
    while tokens and tokens[0] == "@USER":
        tokens.pop(0)
    return " ".join(tokens)


def run_b0(fitted: Fitted, case: dict) -> tuple[AgentOutput, float | None]:
    return AgentOutput(case["case_id"], "b0", fitted.majority_intent, False, "none",
                       "B0 never escalates", None, B0_REPLY, None), None


def run_b1(fitted: Fitted, case: dict) -> tuple[AgentOutput, float | None]:
    query = fitted.vectorizer.transform([index.embedding_text(case["customer_text"])])
    intent = str(fitted.classifier.predict(query)[0])
    similarities = (fitted.matrix @ query.T).toarray().ravel()
    row = int(np.argmax(similarities))  # the first maximum, so the smallest case_id wins a tie
    neighbour = fitted.cases[row]
    match = B1_ESCALATE.search(case["customer_text"])
    if match is None:
        escalate, code, text = False, "none", "no escalation keyword in the message"
    else:
        escalate, code, text = True, "keyword", f"escalation keyword in the message: \"{match.group(0)}\""
    output = AgentOutput(case["case_id"], "b1", intent, escalate, code, text, None,
                         without_leading_handles(neighbour["brand_reply"]), neighbour["case_id"])
    return output, float(similarities[row])


def run_baselines(cases: list[dict]) -> None:
    """B0, then B1, over the cases under one run id: one trace per case each."""
    fitted = fit()
    run_id = new_run_id()
    for system, run_one in (("b0", run_b0), ("b1", run_b1)):
        start_all = time.perf_counter()
        for case in cases:
            start = time.perf_counter()
            output, similarity = run_one(fitted, case)
            write_trace(BaselineTrace(run_id, system, case["case_id"], output, similarity,
                                      time.perf_counter() - start))
        print(f"{system}: {len(cases)} cases in {time.perf_counter() - start_all:.2f} s, zero LLM calls", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B0 and B1 on golden cases and append one trace per case.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N golden cases (default: all 150)")
    args = parser.parse_args()
    run_baselines(golden_cases(args.limit))


if __name__ == "__main__":
    main()
