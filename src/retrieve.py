"""Stage 5 (retrieve), part 2: up to K precedents of the predicted intent, none below a similarity floor.

When even the best precedent of that intent is below the floor, nothing is close enough to ground a reply on:
retrieve returns no precedents, and the escalation ladder treats that as a reason to escalate (layer 3).
"""

import time
from dataclasses import dataclass

import numpy as np

from src import index, taxonomy

K = 5
# A default set before any golden retrieval was looked at, not tuned. The operating curve (REPORT.md 3.2)
# sweeps the escalation threshold upward from here; below it a draft never sees the precedent.
SIMILARITY_FLOOR = 0.50


@dataclass
class Hit:
    precedent: index.Precedent
    similarity: float


@dataclass
class Retrieval:
    hits: list[Hit]  # best first; empty when the best precedent of the intent is below the floor
    max_sim: float  # the best similarity within the intent, kept even when it is below the floor
    floor: float
    latency_s: float


def embed_cases(cases: list[dict]) -> np.ndarray:
    """One query vector per case, from the same text the index embeds."""
    return taxonomy.embed([index.embedding_text(case["customer_text"]) for case in cases])


def retrieve(built: index.Index, query: np.ndarray, intent: str) -> Retrieval:
    """The K most similar precedents of this intent that reach the floor."""
    start = time.perf_counter()
    found = index.search(built, query, K, intent)
    max_sim = found[0][1] if found else 0.0  # an intent with no precedent at all has nothing to ground on
    hits = [Hit(precedent, similarity) for precedent, similarity in found if similarity >= SIMILARITY_FLOOR]
    return Retrieval(hits, max_sim, SIMILARITY_FLOOR, time.perf_counter() - start)
