"""Token tags -> character spans with a span score (min or mean token probability); forced label probabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from note_deid.labels import map_label, strip_bio_prefix
from note_deid.schema import Span
from note_deid.tc.encode import OUTSIDE, Offsets


def decode_tags(
    tags: Sequence[str],
    token_offsets: Offsets,
    probs: Sequence[float] | None = None,
    text: str | None = None,
    source: str = "tc",
    score: str = "min",
) -> list[Span]:
    """Tolerant BIO/BIOES decoding: ``B``/``S`` start a span, ``I``/``E`` continue one with the same label (or start
    a new one if none is open), a label change or ``O`` closes. Special tokens (start == end) are skipped."""
    spans: list[Span] = []
    cur_label: str | None = None
    cur_start = cur_end = 0
    cur_probs: list[float] = []

    def close() -> None:
        nonlocal cur_label
        if cur_label is not None:
            sc = None
            if cur_probs:
                sc = min(cur_probs) if score == "min" else sum(cur_probs) / len(cur_probs)
            spans.append(Span(cur_start, cur_end, cur_label, text[cur_start:cur_end] if text else "", source, sc))
        cur_label = None
        cur_probs.clear()

    for i, (tag, (ts, te)) in enumerate(zip(tags, token_offsets, strict=True)):
        if ts == te:
            continue
        p = probs[i] if probs is not None else None
        if tag == OUTSIDE:
            close()
            continue
        prefix, label = tag[:1], strip_bio_prefix(tag)
        if prefix in ("B", "S") or cur_label != label:
            close()
            cur_label, cur_start, cur_end = label, ts, te
        else:
            cur_end = te
        if p is not None:
            cur_probs.append(p)
        if prefix in ("E", "S"):
            close()
    close()
    return spans


def span_label_prob(
    dist: Sequence[Sequence[float]],
    token_offsets: Offsets,
    id2label: Mapping[int, str],
    start: int,
    end: int,
    label: str,
    label_map: Mapping[str, str | None] | None = None,
    reduce: str = "mean",
) -> float:
    """Forced-decoding probability of ``label`` over ``[start, end)`` (H3 rule (d), framework-designs.md Appendix C).

    For every token overlapping the extent, sum the probability mass on all tags of that label (any BIO/BIOES prefix;
    model labels mapped to the canonical taxonomy through ``label_map`` when given), then reduce with the mean
    (default) or the minimum over the tokens. Returns 0.0 when the model has no tag for the label or no token
    overlaps the extent. ``dist[i]`` is the class distribution of token ``i``; special tokens (empty offsets) are
    skipped.
    """
    targets: set[int] = set()
    for i, tag in id2label.items():
        if tag == OUTSIDE:
            continue
        base = strip_bio_prefix(tag)
        canonical = map_label(base, label_map) if label_map is not None else base
        if canonical == label:
            targets.add(int(i))
    if not targets:
        return 0.0
    masses: list[float] = []
    for row, (ts, te) in zip(dist, token_offsets, strict=True):
        if ts == te or te <= start or ts >= end:
            continue
        masses.append(sum(row[i] for i in targets if i < len(row)))
    if not masses:
        return 0.0
    return min(masses) if reduce == "min" else sum(masses) / len(masses)
