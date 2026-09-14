"""LLM detector: guideline-prompted extraction in formats F1-F4, the H5 editor pass, and the H3 verifier."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from note_deid.labels import I2B2_2014_SUBTYPES
from note_deid.llm.align import AlignStats, align_items, align_segments, align_tagged
from note_deid.llm.client import LLMCall, LLMClient
from note_deid.llm.formats import (
    FORMATS,
    build_editor_messages,
    build_messages,
    build_segment_messages,
    build_verifier_messages,
    load_guideline,
    segment_text,
)
from note_deid.llm.parse import ParseError, parse_json_object, parse_segments, parse_span_items
from note_deid.schema import Doc, Span


@dataclass
class DetectResult:
    spans: list[Span]
    stats: AlignStats
    raw: str
    calls: list[LLMCall] = field(default_factory=list)
    parse_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"n_spans": len(self.spans), "parse_failed": self.parse_failed, **self.stats.to_dict()}


class LLMDetector:
    name = "llm"

    def __init__(
        self,
        client: LLMClient,
        alias: str,
        fmt: str = "json_strings",
        guideline: str | None = None,
        guideline_path: str | None = None,
        labels: Sequence[str] = I2B2_2014_SUBTYPES,
        occurrences: str = "all",
        fuzzy_threshold: float = 90.0,
        segment_chars: int = 1500,
        max_tokens: int | None = None,
        mock_response: str | None = None,
    ) -> None:
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
        self.client = client
        self.alias = alias
        self.fmt = fmt
        self.guideline = load_guideline(guideline_path, guideline)
        self.labels = list(labels)
        self.occurrences = occurrences
        self.fuzzy_threshold = fuzzy_threshold
        self.segment_chars = segment_chars
        self.max_tokens = max_tokens
        self.mock_response = mock_response
        self.results: dict[str, DetectResult] = {}

    def _complete(self, messages: list[dict[str, str]], json_mode: bool) -> tuple[str, LLMCall]:
        r = self.client.complete(
            self.alias, messages, json_mode=json_mode, max_tokens=self.max_tokens, mock_response=self.mock_response
        )
        return r.text, r.call

    def detect_one(self, doc: Doc) -> DetectResult:
        stats = AlignStats()
        text = doc.text
        if self.fmt in ("json_strings", "json_anchored"):
            raw, call = self._complete(build_messages(self.fmt, text, self.guideline, self.labels), json_mode=True)
            try:
                items = parse_span_items(raw)
            except ParseError:
                res = DetectResult([], stats, raw, [call], parse_failed=True)
                self.results[doc.doc_id] = res
                return res
            items = [it for it in items if it["label"] in self.labels] if self.labels else items
            spans = align_items(text, items, self.occurrences, self.fuzzy_threshold, stats)
        elif self.fmt == "inline_tags":
            raw, call = self._complete(build_messages(self.fmt, text, self.guideline, self.labels), json_mode=False)
            spans, ok = align_tagged(text, raw, stats)
            if not ok:
                res = DetectResult([], stats, raw, [call], parse_failed=True)
                self.results[doc.doc_id] = res
                return res
        else:  # segments
            segs = segment_text(text, self.segment_chars)
            raw, call = self._complete(build_segment_messages(text, segs, self.guideline, self.labels), json_mode=False)
            spans = align_segments(text, segs, parse_segments(raw), stats)
        spans = [s for s in spans if not self.labels or s.label in self.labels]
        res = DetectResult(spans, stats, raw, [call])
        self.results[doc.doc_id] = res
        return res

    def detect(self, docs: Iterable[Doc]) -> list[list[Span]]:
        return [self.detect_one(d).spans for d in docs]

    # -- H5 editor -----------------------------------------------------------------------------------------------
    def edit(self, doc: Doc, pretagged: Sequence[Span], focus_labels: Sequence[str] = ()) -> DetectResult:
        stats = AlignStats()
        raw, call = self._complete(
            build_editor_messages(doc.text, pretagged, self.guideline, self.labels, focus_labels), json_mode=False
        )
        spans, ok = align_tagged(doc.text, raw, stats)
        return DetectResult(spans if ok else [], stats, raw, [call], parse_failed=not ok)

    # -- H3 verifier ---------------------------------------------------------------------------------------------
    def verify(self, doc: Doc, span: Span) -> bool:
        raw, _ = self._complete(build_verifier_messages(doc.text, span, self.guideline), json_mode=True)
        try:
            obj = parse_json_object(raw)
        except ParseError:
            return False
        return bool(obj.get("is_phi", False))
