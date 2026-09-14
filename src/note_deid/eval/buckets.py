"""Aggregate per-label results by training-frequency bucket (framework-designs.md §2)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from note_deid.eval.metrics import PRF, relabel
from note_deid.labels import frequency_bucket
from note_deid.schema import Doc


def label_support(docs: Iterable[Doc], level: str = "subtype") -> Counter[str]:
    """Gold span count per label over ``docs`` (use on the TC training split to get n_l)."""
    c: Counter[str] = Counter()
    for d in docs:
        for s in d.spans:
            c[relabel(s, level).label] += 1
    return c


def by_bucket(per_label: Mapping[str, PRF], train_counts: Mapping[str, int]) -> dict[str, PRF]:
    """Sum per-label counts into buckets of their training frequency (labels absent from training -> bucket '0')."""
    out: dict[str, PRF] = {}
    for label, prf in per_label.items():
        b = frequency_bucket(int(train_counts.get(label, 0)))
        out[b] = out.get(b, PRF()) + prf
    order = ["0", "1-10", "11-50", "51-200", ">200"]
    return {b: out[b] for b in order if b in out}
