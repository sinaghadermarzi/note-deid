"""Selective LLM invocation (H7): a per-note risk score from token-classifier uncertainty and surface cues, used to
decide which notes (or segments) are worth an LLM call."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from note_deid.schema import Span

RARE_CUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "url": re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    "phone": re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "fax_word": re.compile(r"\bfax\b", re.IGNORECASE),
    "ip": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "id_like": re.compile(r"\b(?=[A-Z0-9-]{6,}\b)(?=[A-Z-]*\d)[A-Z0-9-]+\b"),
    "zip": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
}
HEADER_LINE = re.compile(
    r"^\s*(patient|name|mrn|dob|date of birth|attending|dictated by|cc:|signed|electronically signed|"
    r"address|phone|tel|fax|email|account|insurance)\b",
    re.IGNORECASE | re.MULTILINE,
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


@dataclass
class RiskFeatures:
    uncertainty_mass: float  # fraction of tokens whose max class probability lies in the uncertain band
    uncovered_cues: int  # regex cue matches not covered by any TC span
    header_lines: int
    capitalized_oov: int  # capitalized mid-sentence words not covered by TC spans and not in the lexicon
    n_chars: int
    tc_span_count: int

    def to_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


DEFAULT_WEIGHTS: dict[str, float] = {
    "uncertainty_mass": 10.0,
    "uncovered_cues": 1.0,
    "header_lines": 0.5,
    "capitalized_oov": 0.2,
    "n_chars": 0.0002,
}


def _covered(start: int, end: int, spans: Sequence[Span]) -> bool:
    return any(s.start < end and start < s.end for s in spans)


def risk_features(
    text: str,
    tc_spans: Sequence[Span],
    token_max_probs: Sequence[float] | None = None,
    lexicon: Collection[str] = (),
    band: tuple[float, float] = (0.2, 0.8),
) -> RiskFeatures:
    if token_max_probs:
        lo, hi = band
        unc = sum(1 for p in token_max_probs if lo <= p <= hi) / len(token_max_probs)
    else:
        unc = 0.0
    cues = 0
    for pat in RARE_CUE_PATTERNS.values():
        for m in pat.finditer(text):
            if not _covered(m.start(), m.end(), tc_spans):
                cues += 1
    headers = len(HEADER_LINE.findall(text))
    oov = 0
    lex = {w.lower() for w in lexicon}
    prev_end = 0
    for m in _WORD.finditer(text):
        word = m.group()
        preceding = text[prev_end : m.start()]
        sentence_start = prev_end == 0 or any(ch in preceding for ch in ".!?\n:")
        prev_end = m.end()
        if word[0].isupper() and not word.isupper() and not sentence_start:
            if word.lower() not in lex and not _covered(m.start(), m.end(), tc_spans):
                oov += 1
    return RiskFeatures(unc, cues, headers, oov, len(text), len(tc_spans))


def risk_score(f: RiskFeatures, weights: Mapping[str, float] | None = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return sum(w[k] * float(getattr(f, k)) for k in w)


def select_for_llm(
    features: Mapping[str, RiskFeatures], rho: float, weights: Mapping[str, float] | None = None
) -> set[str]:
    """Document ids whose risk score exceeds ``rho``."""
    return {doc_id for doc_id, f in features.items() if risk_score(f, weights) > rho}
