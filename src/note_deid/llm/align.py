"""Grounding LLM outputs to character offsets: exact / whitespace-normalized / anchored / fuzzy search for string
items, and diff validation for tagged text (rejecting outputs that changed non-tag characters)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from note_deid.llm.parse import parse_inline_tags
from note_deid.schema import Span

SOURCE = "llm"


@dataclass
class AlignStats:
    exact: int = 0
    whitespace: int = 0
    anchored: int = 0
    fuzzy: int = 0
    failed: int = 0
    failed_items: list[dict] = field(default_factory=list)
    rejected_outputs: int = 0

    def to_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if k != "failed_items"}


def find_all(text: str, needle: str) -> list[int]:
    out, start = [], 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            return out
        out.append(i)
        start = i + 1


def find_whitespace_normalized(text: str, needle: str) -> list[tuple[int, int]]:
    parts = needle.split()
    if not parts:
        return []
    pat = re.compile(r"\s+".join(re.escape(p) for p in parts))
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def align_items(
    text: str,
    items: Sequence[Mapping[str, object]],
    occurrences: str = "all",
    fuzzy_threshold: float = 90.0,
    stats: AlignStats | None = None,
) -> list[Span]:
    """Ground ``{"text", "label", ["before"]}`` items. Exact matches first (all occurrences, or the one whose left
    context ends with ``before``), then whitespace-normalized, then fuzzy (rapidfuzz partial alignment)."""
    stats = stats if stats is not None else AlignStats()
    spans: dict[tuple[int, int], Span] = {}

    def add(start: int, end: int, label: str, score: float) -> None:
        key = (start, end)
        if key not in spans or (spans[key].score or 0) < score:
            spans[key] = Span(start, end, label, text[start:end], SOURCE, score)

    for it in items:
        needle, label = str(it["text"]), str(it["label"])
        before = it.get("before")
        positions = find_all(text, needle)
        if positions:
            if before and isinstance(before, str) and before.strip():
                anchored = [
                    p for p in positions if _norm(text[max(0, p - len(before) - 5) : p]).endswith(_norm(before))
                ]
                if anchored:
                    add(anchored[0], anchored[0] + len(needle), label, 1.0)
                    stats.anchored += 1
                    continue
            chosen = positions if occurrences == "all" else positions[:1]
            for p in chosen:
                add(p, p + len(needle), label, 1.0)
            stats.exact += 1
            continue
        ws = find_whitespace_normalized(text, needle)
        if ws:
            for s, e in ws if occurrences == "all" else ws[:1]:
                add(s, e, label, 0.95)
            stats.whitespace += 1
            continue
        try:
            from rapidfuzz import fuzz

            res = fuzz.partial_ratio_alignment(needle, text)
        except ImportError:  # pragma: no cover
            res = None
        if res is not None and res.score >= fuzzy_threshold and res.dest_end > res.dest_start:
            s, e = res.dest_start, res.dest_end
            # snap to word boundaries
            while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
                s -= 1
            while e < len(text) and text[e - 1].isalnum() and text[e].isalnum():
                e += 1
            add(s, e, label, res.score / 100.0)
            stats.fuzzy += 1
            continue
        stats.failed += 1
        stats.failed_items.append(dict(it))
    return sorted(spans.values(), key=lambda s: (s.start, s.end))


def map_offsets_ignoring_whitespace(original: str, other: str) -> list[int] | None:
    """For each index in ``other`` the corresponding index in ``original`` when the two differ only in whitespace;
    ``None`` if the non-whitespace characters differ."""
    o_idx = [i for i, ch in enumerate(original) if not ch.isspace()]
    t_idx = [i for i, ch in enumerate(other) if not ch.isspace()]
    if len(o_idx) != len(t_idx) or any(original[i] != other[j] for i, j in zip(o_idx, t_idx, strict=True)):
        return None
    mapping = [0] * (len(other) + 1)
    k = 0
    for j in range(len(other)):
        if k < len(t_idx) and j == t_idx[k]:
            mapping[j] = o_idx[k]
            k += 1
        else:  # whitespace in other: map to the next non-space char in original (or end)
            mapping[j] = o_idx[k] if k < len(o_idx) else len(original)
    mapping[len(other)] = len(original)
    return mapping


def align_tagged(text: str, tagged_output: str, stats: AlignStats | None = None) -> tuple[list[Span], bool]:
    """Spans from a tagged rewrite of ``text``. Accepted only if the untagged output equals ``text`` (exactly, or up to
    whitespace); otherwise ``([], False)`` and the output counts as rejected."""
    stats = stats if stats is not None else AlignStats()
    plain, raw = parse_inline_tags(tagged_output.strip())
    if plain == text:
        spans = [Span(s, e, lab, text[s:e], SOURCE, 1.0) for s, e, lab in raw]
        stats.exact += len(spans)
        return spans, True
    mapping = map_offsets_ignoring_whitespace(text, plain)
    if mapping is None:
        stats.rejected_outputs += 1
        return [], False
    spans = []
    for s, e, lab in raw:
        ms, me = mapping[s], mapping[e - 1] + 1 if e > s else mapping[s]
        # trim whitespace at the edges of the mapped extent
        while ms < me and text[ms].isspace():
            ms += 1
        while me > ms and text[me - 1].isspace():
            me -= 1
        if me > ms:
            spans.append(Span(ms, me, lab, text[ms:me], SOURCE, 0.95))
    stats.whitespace += len(spans)
    return spans, True


def align_segments(
    text: str,
    segs: Sequence[tuple[int, int]],
    outputs: Mapping[int, str],
    stats: AlignStats | None = None,
) -> list[Span]:
    """Per-segment tagged outputs (1-based numbering) -> document spans; a rejected segment contributes nothing."""
    stats = stats if stats is not None else AlignStats()
    spans: list[Span] = []
    for i, (s, e) in enumerate(segs):
        out = outputs.get(i + 1)
        if out is None:
            stats.rejected_outputs += 1
            continue
        seg_text = text[s:e]
        seg_spans, ok = align_tagged(seg_text.replace("\n", " "), out, stats)
        if not ok:
            continue
        for sp in seg_spans:
            spans.append(Span(s + sp.start, s + sp.end, sp.label, text[s + sp.start : s + sp.end], SOURCE, sp.score))
    return spans
