"""Paired bootstrap over documents and Holm correction (framework-designs.md §5)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from note_deid.eval.metrics import PRF, DocCounts


def micro_metric(counts: Sequence[DocCounts], metric: str = "recall") -> float:
    total = sum((c.total for c in counts), PRF())
    if metric == "recall":
        return total.recall
    if metric == "precision":
        return total.precision
    if metric == "f1":
        return total.f1
    if metric == "leak_rate":
        return sum(1 for c in counts if c.leaked) / len(counts) if counts else 0.0
    raise ValueError(f"unknown metric {metric!r}")


@dataclass
class BootstrapResult:
    metric: str
    observed_a: float
    observed_b: float
    delta: float  # a - b on the full sample
    ci_low: float
    ci_high: float
    p_value: float  # two-sided bootstrap p-value for delta != 0
    n_resamples: int
    n_docs: int
    seed: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def paired_bootstrap(
    counts_a: Sequence[DocCounts],
    counts_b: Sequence[DocCounts],
    metric: str = "recall",
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Resample documents with replacement (same indices for both systems) and recompute the micro metric."""
    if [c.doc_id for c in counts_a] != [c.doc_id for c in counts_b]:
        raise ValueError("counts_a and counts_b must cover the same documents in the same order")
    n = len(counts_a)
    if n == 0:
        raise ValueError("no documents")
    obs_a, obs_b = micro_metric(counts_a, metric), micro_metric(counts_b, metric)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [counts_a[i] for i in idx]
        sb = [counts_b[i] for i in idx]
        deltas.append(micro_metric(sa, metric) - micro_metric(sb, metric))
    deltas.sort()
    lo = deltas[int((alpha / 2) * (n_resamples - 1))]
    hi = deltas[int((1 - alpha / 2) * (n_resamples - 1))]
    frac_le = sum(1 for d in deltas if d <= 0) / n_resamples
    frac_ge = sum(1 for d in deltas if d >= 0) / n_resamples
    p = min(1.0, 2 * min(frac_le, frac_ge))
    return BootstrapResult(metric, obs_a, obs_b, obs_a - obs_b, lo, hi, p, n_resamples, n, seed)


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1), in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p_values[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted
