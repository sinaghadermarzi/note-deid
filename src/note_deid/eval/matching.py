"""Span matching semantics (i2b2-2014 style).

- ``strict``: identical offsets and label.
- ``relaxed``: character overlap and identical label.
- ``relaxed_typeless``: character overlap only (privacy view: a date masked as NAME is still masked).

Matching is greedy and one-to-one: gold spans are visited in document order and each takes the best unmatched
prediction (exact extent preferred, then largest overlap). A prediction that covers two gold spans therefore matches
only one of them; the other is a false negative. This is the conservative reading of "relaxed".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from note_deid.schema import Span

MatchMode = Literal["strict", "relaxed", "relaxed_typeless"]
MATCH_MODES: tuple[str, ...] = ("strict", "relaxed", "relaxed_typeless")


@dataclass
class MatchResult:
    mode: str
    tp: list[tuple[Span, Span]]  # (gold, prediction) pairs
    fn: list[Span]  # gold spans without a match
    fp: list[Span]  # predictions without a match

    @property
    def counts(self) -> tuple[int, int, int]:
        """(tp, fp, fn)"""
        return len(self.tp), len(self.fp), len(self.fn)


def overlap_length(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def spans_match(gold: Span, pred: Span, mode: str) -> bool:
    if mode == "strict":
        return gold.start == pred.start and gold.end == pred.end and gold.label == pred.label
    if mode == "relaxed":
        return gold.label == pred.label and overlap_length(gold, pred) > 0
    if mode == "relaxed_typeless":
        return overlap_length(gold, pred) > 0
    raise ValueError(f"unknown match mode {mode!r}; expected one of {MATCH_MODES}")


def span_key(s: Span) -> tuple[int, int, str]:
    """Identity of a span for set operations (offsets + label; source/score ignored)."""
    return (s.start, s.end, s.label)


def match_spans(gold: list[Span], pred: list[Span], mode: str = "strict") -> MatchResult:
    if mode not in MATCH_MODES:
        raise ValueError(f"unknown match mode {mode!r}; expected one of {MATCH_MODES}")
    gold_sorted = sorted(gold, key=span_key)
    pred_sorted = sorted(pred, key=span_key)
    used: set[int] = set()
    tp: list[tuple[Span, Span]] = []
    fn: list[Span] = []
    for g in gold_sorted:
        best_j = -1
        best_score = (-1, -1)
        for j, p in enumerate(pred_sorted):
            if j in used or not spans_match(g, p, mode):
                continue
            score = (int(g.start == p.start and g.end == p.end), overlap_length(g, p))
            if score > best_score:
                best_score, best_j = score, j
        if best_j < 0:
            fn.append(g)
        else:
            used.add(best_j)
            tp.append((g, pred_sorted[best_j]))
    fp = [p for j, p in enumerate(pred_sorted) if j not in used]
    return MatchResult(mode, tp, fn, fp)
