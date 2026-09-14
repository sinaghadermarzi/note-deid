"""Entity-, token- and document-level metrics with i2b2-2014 semantics.

``evaluate`` is the entry point; ``evaluate_docs`` returns per-document counts that the bootstrap and bucket
analyses resample or regroup. Labels can be evaluated at ``subtype`` (28 leaf labels), ``category`` (7) or
``binary`` (PHI / not) level, optionally restricted to the HIPAA Safe Harbor view (``hipaa_only``: HIPAA subtypes
only, AGE counted only when the mention is >= 90).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from note_deid.eval.matching import MATCH_MODES, match_spans
from note_deid.labels import HIPAA_SUBTYPES, SUBTYPE_TO_CATEGORY
from note_deid.schema import Doc, Span

Level = Literal["subtype", "category", "binary"]
LEVELS: tuple[str, ...] = ("subtype", "category", "binary")
BINARY_LABEL = "PHI"

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_INT_RE = re.compile(r"\d{1,3}")


@dataclass(frozen=True)
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def support(self) -> int:
        """Number of gold spans."""
        return self.tp + self.fn

    def __add__(self, other: PRF) -> PRF:
        return PRF(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


# -- label views ---------------------------------------------------------------------------------------------------
def relabel(span: Span, level: str) -> Span:
    if level == "subtype":
        return span
    if level == "category":
        label = SUBTYPE_TO_CATEGORY.get(span.label, span.label)
    elif level == "binary":
        label = BINARY_LABEL
    else:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    return Span(span.start, span.end, label, span.text, span.source, span.score)


def is_hipaa_span(span: Span, text: str) -> bool:
    """HIPAA Safe Harbor view: HIPAA subtypes, and AGE only when the mentioned age is >= 90."""
    if span.label == "AGE":
        m = _INT_RE.search(text[span.start : span.end] or span.text)
        return bool(m) and int(m.group()) >= 90
    return span.label in HIPAA_SUBTYPES


def prepare_spans(spans: Iterable[Span], text: str, level: str, hipaa_only: bool) -> list[Span]:
    kept = [s for s in spans if not hipaa_only or is_hipaa_span(s, text)]
    return [relabel(s, level) for s in kept]


# -- per-document counts -------------------------------------------------------------------------------------------
@dataclass
class DocCounts:
    doc_id: str
    per_label: dict[str, PRF]
    token: PRF | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> PRF:
        return sum(self.per_label.values(), PRF())

    @property
    def leaked(self) -> bool:
        return self.total.fn > 0


def token_prf(text: str, gold: Sequence[Span], pred: Sequence[Span]) -> PRF:
    """Binary token-level counts: a token is PHI if it overlaps any span."""
    tokens = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]

    def covered(spans: Sequence[Span]) -> set[int]:
        out: set[int] = set()
        for s in spans:
            for i, (ts, te) in enumerate(tokens):
                if ts < s.end and s.start < te:
                    out.add(i)
        return out

    g, p = covered(gold), covered(pred)
    return PRF(len(g & p), len(p - g), len(g - p))


def doc_counts(
    gold: Doc,
    pred_spans: Sequence[Span],
    mode: str = "strict",
    level: str = "subtype",
    hipaa_only: bool = False,
    token_level: bool = True,
) -> DocCounts:
    if mode not in MATCH_MODES:
        raise ValueError(f"unknown match mode {mode!r}")
    g = prepare_spans(gold.spans, gold.text, level, hipaa_only)
    p = prepare_spans(pred_spans, gold.text, level, hipaa_only)
    res = match_spans(g, p, mode)
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for gs, _ in res.tp:
        acc[gs.label][0] += 1
    for s in res.fp:
        acc[s.label][1] += 1
    for s in res.fn:
        acc[s.label][2] += 1
    per_label = {lab: PRF(*v) for lab, v in sorted(acc.items())}
    tok = token_prf(gold.text, g, p) if token_level else None
    return DocCounts(gold.doc_id, per_label, tok, dict(gold.meta))


Predictions = Mapping[str, Sequence[Span]] | Iterable[Doc]


def _as_prediction_map(predictions: Predictions) -> dict[str, Sequence[Span]]:
    if isinstance(predictions, Mapping):
        return dict(predictions)
    return {d.doc_id: d.spans for d in predictions}


def evaluate_docs(
    gold_docs: Iterable[Doc],
    predictions: Predictions,
    mode: str = "strict",
    level: str = "subtype",
    hipaa_only: bool = False,
    token_level: bool = True,
) -> list[DocCounts]:
    pred_map = _as_prediction_map(predictions)
    return [doc_counts(d, pred_map.get(d.doc_id, []), mode, level, hipaa_only, token_level) for d in gold_docs]


# -- aggregate report ----------------------------------------------------------------------------------------------
@dataclass
class Report:
    mode: str
    level: str
    hipaa_only: bool
    n_docs: int
    per_label: dict[str, PRF]
    micro: PRF
    macro_precision: float
    macro_recall: float
    macro_f1: float
    doc_leak_rate: float  # fraction of all docs with >= 1 false negative
    docs_with_gold: int
    leak_rate_among_docs_with_gold: float
    token: PRF | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "level": self.level,
            "hipaa_only": self.hipaa_only,
            "n_docs": self.n_docs,
            "micro": self.micro.to_dict(),
            "macro": {
                "precision": round(self.macro_precision, 6),
                "recall": round(self.macro_recall, 6),
                "f1": round(self.macro_f1, 6),
            },
            "doc_leak_rate": round(self.doc_leak_rate, 6),
            "docs_with_gold": self.docs_with_gold,
            "leak_rate_among_docs_with_gold": round(self.leak_rate_among_docs_with_gold, 6),
            "token": self.token.to_dict() if self.token else None,
            "per_label": {k: v.to_dict() for k, v in self.per_label.items()},
        }


def aggregate(counts: Sequence[DocCounts], mode: str, level: str, hipaa_only: bool) -> Report:
    per_label: dict[str, PRF] = defaultdict(PRF)
    token = PRF()
    has_token = any(c.token is not None for c in counts)
    for c in counts:
        for lab, prf in c.per_label.items():
            per_label[lab] = per_label[lab] + prf
        if c.token is not None:
            token = token + c.token
    per_label = dict(sorted(per_label.items()))
    micro = sum(per_label.values(), PRF())
    supported = [v for v in per_label.values() if v.support > 0]
    n = len(supported)
    macro_p = sum(v.precision for v in supported) / n if n else 0.0
    macro_r = sum(v.recall for v in supported) / n if n else 0.0
    macro_f = sum(v.f1 for v in supported) / n if n else 0.0
    n_docs = len(counts)
    leaked = sum(1 for c in counts if c.leaked)
    with_gold = [c for c in counts if c.total.support > 0]
    return Report(
        mode=mode,
        level=level,
        hipaa_only=hipaa_only,
        n_docs=n_docs,
        per_label=per_label,
        micro=micro,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
        doc_leak_rate=leaked / n_docs if n_docs else 0.0,
        docs_with_gold=len(with_gold),
        leak_rate_among_docs_with_gold=(sum(1 for c in with_gold if c.leaked) / len(with_gold)) if with_gold else 0.0,
        token=token if has_token else None,
    )


def evaluate(
    gold_docs: Iterable[Doc],
    predictions: Predictions,
    mode: str = "strict",
    level: str = "subtype",
    hipaa_only: bool = False,
    token_level: bool = True,
) -> Report:
    gold_list = list(gold_docs)
    counts = evaluate_docs(gold_list, predictions, mode, level, hipaa_only, token_level)
    return aggregate(counts, mode, level, hipaa_only)
