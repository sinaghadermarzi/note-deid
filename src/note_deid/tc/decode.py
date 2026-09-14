"""Token tags -> character spans with a span score (min or mean token probability)."""

from __future__ import annotations

from collections.abc import Sequence

from note_deid.labels import strip_bio_prefix
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
