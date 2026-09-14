"""Inference-time fusion policies (framework-designs.md §3.1, Appendix C).

All policies take the per-system outputs ``{system_name: [Span, ...]}`` for one document and return the fused span
list. Spans keep their ``source`` (joined with ``+`` when merged) and ``score``. Systems are named by convention
``"tc"`` (token classifier, spans carry probabilities) and ``"llm"``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

from note_deid.eval.matching import overlap_length, span_key
from note_deid.labels import OPF_UNSUPPORTED_SUBTYPES
from note_deid.schema import Span

Outputs = Mapping[str, Sequence[Span]]


def _with_text(s: Span, text: str | None) -> Span:
    if text is None or s.text:
        return s
    return Span(s.start, s.end, s.label, text[s.start : s.end], s.source, s.score)


def _merge_two(a: Span, b: Span, text: str | None) -> Span:
    sources = "+".join(sorted(set(a.source.split("+")) | set(b.source.split("+"))))
    scores = [x for x in (a.score, b.score) if x is not None]
    start, end = min(a.start, b.start), max(a.end, b.end)
    return Span(start, end, a.label, text[start:end] if text else "", sources, max(scores) if scores else None)


def merge_overlapping(spans: Sequence[Span], text: str | None = None) -> list[Span]:
    """Merge overlapping spans that carry the same label into their union extent (max score, joined sources)."""
    out: list[Span] = []
    for s in sorted(spans, key=span_key):
        s = _with_text(s, text)
        for i, prev in enumerate(out):
            if prev.label == s.label and overlap_length(prev, s) > 0:
                out[i] = _merge_two(prev, s, text)
                break
        else:
            out.append(s)
    return sorted(out, key=span_key)


def _source_rank(source: str, priority: Sequence[str]) -> int:
    ranks = [priority.index(p) for p in source.split("+") if p in priority]
    return min(ranks) if ranks else len(priority)


def resolve_conflicts(
    spans: Sequence[Span],
    priority: Sequence[str] = ("tc", "llm"),
    min_score: Mapping[str, float] | None = None,
) -> list[Span]:
    """Among overlapping spans with *different* labels keep one: the span whose source ranks first in ``priority``
    (a source whose score is below ``min_score[source]`` is demoted to the end), ties broken by higher score then
    longer extent."""
    min_score = min_score or {}

    def rank(s: Span) -> tuple[int, float, int]:
        r = _source_rank(s.source, priority)
        for src in s.source.split("+"):
            if src in min_score and (s.score is None or s.score < min_score[src]):
                r = len(priority)
        return (r, -(s.score or 0.0), -s.length)

    kept: list[Span] = []
    for s in sorted(spans, key=rank):
        if all(overlap_length(s, k) == 0 or k.label == s.label for k in kept):
            kept.append(s)
    return sorted(kept, key=span_key)


def union(
    outputs: Outputs,
    text: str | None = None,
    priority: Sequence[str] = ("tc", "llm"),
    min_score: Mapping[str, float] | None = None,
) -> list[Span]:
    """H1: every span from every system; same-label overlaps merged, label conflicts resolved by ``priority``."""
    merged = merge_overlapping([s for spans in outputs.values() for s in spans], text)
    return resolve_conflicts(merged, priority, min_score)


def label_routed(
    outputs: Outputs,
    train_counts: Mapping[str, int],
    tau: int = 10,
    tc: str = "tc",
    llm: str = "llm",
    unsupported: Collection[str] = OPF_UNSUPPORTED_SUBTYPES,
    both: Collection[str] = (),
    text: str | None = None,
) -> list[Span]:
    """H2: per label take the TC output when the label is well represented in TC training data (count >= tau) and
    supported by the TC label space, the LLM output otherwise, and the union for labels listed in ``both``."""
    chosen: list[Span] = []
    for s in outputs.get(tc, []):
        if s.label in both or (s.label not in unsupported and train_counts.get(s.label, 0) >= tau):
            chosen.append(s)
    for s in outputs.get(llm, []):
        if s.label in both or s.label in unsupported or train_counts.get(s.label, 0) < tau:
            chosen.append(s)
    return resolve_conflicts(merge_overlapping(chosen, text), priority=(tc, llm))


@dataclass
class GateConfig:
    theta_hi: float = 0.9  # TC-only span accepted outright at or above this score
    theta_lo: float = 0.3  # LLM-only span accepted if the TC's forced probability for it is at least this
    tau: int = 10  # labels with fewer TC training instances are trusted to the LLM
    unsupported: frozenset[str] = OPF_UNSUPPORTED_SUBTYPES
    tc: str = "tc"
    llm: str = "llm"


@dataclass
class Decision:
    span: Span
    rule: str
    accepted: bool


@dataclass
class GateResult:
    spans: list[Span]
    decisions: list[Decision] = field(default_factory=list)
    verifier_calls: int = 0


def confidence_gated(
    outputs: Outputs,
    train_counts: Mapping[str, int],
    cfg: GateConfig | None = None,
    verifier: Callable[[Span], bool] | None = None,
    tc_forced_prob: Callable[[Span], float] | None = None,
    text: str | None = None,
) -> GateResult:
    """H3 arbitration (Appendix C).

    - overlap between a TC span and an LLM span -> accept (TC extent; TC label if its score >= theta_lo, else LLM label)
    - TC-only: accept if score >= theta_hi, else ask ``verifier`` (drop if none)
    - LLM-only: accept if the label is TC-unsupported or rare (< tau); else accept if ``tc_forced_prob`` >= theta_lo,
      else ask ``verifier`` (drop if none)
    """
    cfg = cfg or GateConfig()
    tc_spans = merge_overlapping(outputs.get(cfg.tc, []), text)
    llm_spans = merge_overlapping(outputs.get(cfg.llm, []), text)
    result = GateResult([])
    matched_llm: set[tuple[int, int, str]] = set()

    def decide(span: Span, rule: str, accepted: bool) -> None:
        result.decisions.append(Decision(span, rule, accepted))
        if accepted:
            result.spans.append(span)

    def ask(span: Span, rule: str) -> None:
        if verifier is None:
            decide(span, rule + ":no-verifier", False)
            return
        result.verifier_calls += 1
        decide(span, rule + ":verified", bool(verifier(span)))

    for t in tc_spans:
        partners = [lspan for lspan in llm_spans if overlap_length(t, lspan) > 0]
        if partners:
            for lspan in partners:
                matched_llm.add(span_key(lspan))
            label = t.label if (t.score is None or t.score >= cfg.theta_lo) else partners[0].label
            agreed = Span(t.start, t.end, label, t.text, f"{cfg.llm}+{cfg.tc}", t.score)
            decide(agreed, "agree", True)
        elif t.score is not None and t.score >= cfg.theta_hi:
            decide(t, "tc-only:high", True)
        else:
            ask(t, "tc-only:low")

    for lspan in llm_spans:
        if span_key(lspan) in matched_llm:
            continue
        if lspan.label in cfg.unsupported or train_counts.get(lspan.label, 0) < cfg.tau:
            decide(lspan, "llm-only:rare-or-unsupported", True)
        elif tc_forced_prob is not None and tc_forced_prob(lspan) >= cfg.theta_lo:
            decide(lspan, "llm-only:tc-forced", True)
        else:
            ask(lspan, "llm-only:frequent")

    result.spans = resolve_conflicts(merge_overlapping(result.spans, text), priority=(cfg.tc, cfg.llm))
    return result
