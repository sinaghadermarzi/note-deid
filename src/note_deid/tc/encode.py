"""Character spans <-> token tags (BIO / BIOES) over tokenizer character offsets, plus window chunking."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from note_deid.schema import Span

Scheme = Literal["bio", "bioes"]
OUTSIDE = "O"
IGNORE_INDEX = -100
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

Offsets = Sequence[tuple[int, int]]


def regex_token_offsets(text: str) -> list[tuple[int, int]]:
    """Whitespace/punctuation tokenizer offsets (for tests and for models without a fast tokenizer)."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def spans_to_token_labels(
    spans: Sequence[Span], token_offsets: Offsets, scheme: Scheme = "bioes", outside: str = OUTSIDE
) -> list[str]:
    """Tag every token from its character offsets. Special tokens (start == end) get ``outside``; overlapping spans
    should be merged beforehand (the first span in document order wins on a contested token)."""
    tags = [outside] * len(token_offsets)
    claimed = [False] * len(token_offsets)
    for s in sorted(spans, key=lambda x: (x.start, x.end)):
        idx = [
            i for i, (ts, te) in enumerate(token_offsets) if te > ts and ts < s.end and s.start < te and not claimed[i]
        ]
        if not idx:
            continue
        for i in idx:
            claimed[i] = True
        if scheme == "bio":
            tags[idx[0]] = f"B-{s.label}"
            for i in idx[1:]:
                tags[i] = f"I-{s.label}"
        elif scheme == "bioes":
            if len(idx) == 1:
                tags[idx[0]] = f"S-{s.label}"
            else:
                tags[idx[0]] = f"B-{s.label}"
                for i in idx[1:-1]:
                    tags[i] = f"I-{s.label}"
                tags[idx[-1]] = f"E-{s.label}"
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
    return tags


def tags_to_ids(tags: Sequence[str], token_offsets: Offsets, label2id: dict[str, int]) -> list[int]:
    """Tag ids with ``IGNORE_INDEX`` on special tokens (offset start == end) so the loss skips them."""
    return [IGNORE_INDEX if ts == te else label2id[t] for t, (ts, te) in zip(tags, token_offsets, strict=True)]


def chunk_windows(n_tokens: int, max_tokens: int, stride: int) -> list[tuple[int, int]]:
    """[start, end) token windows of at most ``max_tokens`` overlapping by ``stride`` tokens."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if n_tokens <= max_tokens:
        return [(0, n_tokens)]
    if stride >= max_tokens:
        raise ValueError("stride must be smaller than max_tokens")
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + max_tokens, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            return windows
        start = end - stride


def merge_window_predictions(
    windows: Sequence[tuple[int, int]], per_window: Sequence[Sequence[tuple[int, float]]], n_tokens: int
) -> list[tuple[int, float]]:
    """Combine per-window (label_id, prob) predictions into one sequence: a token covered by several windows takes
    the prediction from the window in which it is farthest from an edge."""
    best: list[tuple[int, int, float]] = [(-1, 0, 0.0)] * n_tokens  # (centrality, label_id, prob)
    for (ws, we), preds in zip(windows, per_window, strict=True):
        for k, (label_id, prob) in enumerate(preds):
            i = ws + k
            centrality = min(k, (we - ws - 1) - k)
            if centrality > best[i][0]:
                best[i] = (centrality, label_id, prob)
    return [(lab, prob) for _, lab, prob in best]
