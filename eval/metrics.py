"""Metric arithmetic, no I/O: per-class and macro F1 (optionally weighted), confusion matrices, weighted means,
reweighting to population shares, and bootstrap intervals, paired ones included. Every bootstrap draws the same
RESAMPLES rows with the same seed, so two statistics on the same cases are always paired."""

import math
from collections import Counter
from typing import Callable

import numpy as np

RESAMPLES = 2000
SEED = 0
Statistic = Callable[[np.ndarray], float]  # a metric computed on an array of case positions


def f1_by_class(truth: list[str], predicted: list[str], classes: list[str],
                weights: list[float] | None = None) -> dict[str, dict[str, float]]:
    """Per class: (weighted) true positives, false positives, false negatives, support and F1."""
    if weights is None:
        weights = [1.0] * len(truth)
    table = {name: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "support": 0.0} for name in classes}
    for actual, guess, weight in zip(truth, predicted, weights):
        table[actual]["support"] += weight
        if actual == guess:
            table[actual]["tp"] += weight
        else:
            table[actual]["fn"] += weight
            table[guess]["fp"] += weight
    for row in table.values():
        denominator = 2 * row["tp"] + row["fp"] + row["fn"]
        row["f1"] = 2 * row["tp"] / denominator if denominator > 0 else 0.0
    return table


def macro_f1(truth: list[str], predicted: list[str], classes: list[str],
             weights: list[float] | None = None) -> float:
    """Mean F1 over the classes that occur in the truth: a class with no support has no F1 to average."""
    table = f1_by_class(truth, predicted, classes, weights)
    supported = [row["f1"] for row in table.values() if row["support"] > 0]
    return sum(supported) / len(supported) if supported else math.nan


def confusion(truth: list[str], predicted: list[str], classes: list[str]) -> list[list[int]]:
    """Rows are the true class and columns the predicted class, both in `classes` order."""
    position = {name: number for number, name in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for actual, guess in zip(truth, predicted):
        matrix[position[actual]][position[guess]] += 1
    return matrix


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / total if total > 0 else math.nan


def reweight(intents: list[str], shares: dict[str, float]) -> list[float]:
    """Per case: its intent's population share over that intent's share of these cases, so the weighted cases
    have the population's mix. An intent absent from these cases cannot be represented at all."""
    counts = Counter(intents)
    return [shares[intent] / (counts[intent] / len(intents)) for intent in intents]


def macro_statistic(truth: list[str], predicted: list[str], classes: list[str],
                    weights: list[float] | None = None) -> Statistic:
    """Macro-F1 on resampled case positions, as a bootstrap statistic."""
    def statistic(rows: np.ndarray) -> float:
        picked_weights = None if weights is None else [weights[row] for row in rows]
        return macro_f1([truth[row] for row in rows], [predicted[row] for row in rows], classes, picked_weights)
    return statistic


def mean_statistic(vector: np.ndarray) -> Statistic:
    """The mean of a per-case vector on resampled positions, skipping NaN (a case with no usable value)."""
    def statistic(rows: np.ndarray) -> float:
        picked = vector[rows]
        picked = picked[~np.isnan(picked)]
        return float(picked.mean()) if picked.size else math.nan
    return statistic


def resamples(n: int) -> np.ndarray:
    """RESAMPLES rows of n case positions drawn with replacement, identical on every call."""
    return np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))


def interval(values: list[float]) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles, ignoring resamples where the statistic is undefined."""
    kept = [value for value in values if not math.isnan(value)]
    if not kept:
        return math.nan, math.nan
    return float(np.percentile(kept, 2.5)), float(np.percentile(kept, 97.5))


def bootstrap_ci(n: int, statistic: Statistic) -> tuple[float, float]:
    """95% percentile interval of the statistic over the resamples."""
    return interval([statistic(rows) for rows in resamples(n)])


def paired_difference(n: int, statistic_a: Statistic, statistic_b: Statistic) -> tuple[float, float, float]:
    """a - b on all cases, and its 95% interval with both statistics computed on the same resampled cases."""
    everything = np.arange(n)
    point = statistic_a(everything) - statistic_b(everything)
    low, high = interval([statistic_a(rows) - statistic_b(rows) for rows in resamples(n)])
    return point, low, high
