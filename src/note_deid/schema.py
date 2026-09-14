"""Unified document/span schema shared by every backend.

Offsets are character offsets into ``Doc.text``; ``end`` is exclusive (Python slice semantics). This matches the
OpenAI Privacy Filter (OPF) JSONL convention and what the i2b2-2014 XML converter will produce.

A ``Span`` records where it came from (``source``: ``gold``, ``llm``, ``tc``, ``hybrid`` ...) and an optional
``score`` (token-classifier span probability, LLM grounding score, ...) so fusion policies can arbitrate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

GOLD = "gold"


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    label: str
    text: str = ""
    source: str = GOLD
    score: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span offsets: start={self.start} end={self.end}")
        if not self.label:
            raise ValueError("span label must be non-empty")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def same_extent(self, other: Span) -> bool:
        return self.start == other.start and self.end == other.end

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["score"] is None:
            d.pop("score")
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Span:
        return cls(
            start=int(d["start"]),
            end=int(d["end"]),
            label=str(d["label"]),
            text=str(d.get("text", "")),
            source=str(d.get("source", GOLD)),
            score=d.get("score"),
        )


@dataclass
class Doc:
    doc_id: str
    text: str
    spans: list[Span] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- integrity -------------------------------------------------------------------------------------------
    def validate(self, check_text: bool = True) -> Doc:
        """Raise ``ValueError`` if any span falls outside the text or disagrees with the text slice."""
        n = len(self.text)
        for s in self.spans:
            if s.end > n:
                raise ValueError(f"{self.doc_id}: span {s} exceeds text length {n}")
            if check_text and s.text and self.text[s.start : s.end] != s.text:
                raise ValueError(
                    f"{self.doc_id}: span text {s.text!r} != document slice {self.text[s.start : s.end]!r}"
                )
        return self

    def fill_span_text(self) -> Doc:
        """Populate ``Span.text`` from the document for spans that lack it."""
        self.spans = [
            s if s.text else Span(s.start, s.end, s.label, self.text[s.start : s.end], s.source, s.score)
            for s in self.spans
        ]
        return self

    def sorted_spans(self) -> list[Span]:
        return sorted(self.spans, key=lambda s: (s.start, -s.end, s.label))

    # -- serialization ---------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "spans": [s.to_dict() for s in self.sorted_spans()],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Doc:
        return cls(
            doc_id=str(d["doc_id"]),
            text=str(d["text"]),
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
            meta=dict(d.get("meta", {})),
        )


# -- JSONL I/O ---------------------------------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> Iterator[Doc]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Doc.from_dict(json.loads(line))


def write_jsonl(path: str | Path, docs: Iterable[Doc]) -> int:
    """Write docs as one JSON object per line; returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


# -- OpenAI Privacy Filter JSONL ---------------------------------------------------------------------------------
# Record format (verified against examples/data/sample_eval_five_examples.jsonl in openai/privacy-filter):
#   {"text": "...", "spans": {"<label>: <value>": [[start, end], ...]}, "info": {"id": "...", "source": "..."}}
def to_opf_record(
    doc: Doc,
    label_map: Mapping[str, str | None] | None = None,
    source: str = "note_deid",
) -> dict[str, Any]:
    """Convert a ``Doc`` to an OPF training/eval record.

    ``label_map`` maps canonical labels to OPF labels; a ``None`` target drops the span (OPF cannot represent it).
    Without a map, labels are written unchanged (only valid when the doc already uses OPF labels).
    """
    spans: dict[str, list[list[int]]] = {}
    for s in doc.sorted_spans():
        label = label_map[s.label] if label_map is not None else s.label
        if label is None:
            continue
        value = doc.text[s.start : s.end]
        spans.setdefault(f"{label}: {value}", []).append([s.start, s.end])
    return {"text": doc.text, "spans": spans, "info": {"id": doc.doc_id, "source": source, "meta": dict(doc.meta)}}


def from_opf_record(rec: Mapping[str, Any], source: str = GOLD) -> Doc:
    text = str(rec["text"])
    spans: list[Span] = []
    for key, offsets in rec.get("spans", {}).items():
        label, sep, _value = key.partition(": ")
        if not sep:
            raise ValueError(f"malformed OPF span key {key!r}; expected '<label>: <value>'")
        for start, end in offsets:
            spans.append(Span(int(start), int(end), label, text[int(start) : int(end)], source))
    info = rec.get("info", {}) or {}
    doc_id = str(info.get("id") or "opf-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12])
    return Doc(doc_id=doc_id, text=text, spans=spans, meta=dict(info.get("meta", {}))).validate()


# -- redaction ---------------------------------------------------------------------------------------------------
def redact(doc: Doc, template: str = "[{label}]") -> str:
    """Replace every span with ``template`` (formatted with the span's label); overlapping spans are merged."""
    out: list[str] = []
    cursor = 0
    for s in doc.sorted_spans():
        if s.start < cursor:  # overlaps the previous span: extend the redaction without a second placeholder
            cursor = max(cursor, s.end)
            continue
        out.append(doc.text[cursor : s.start])
        out.append(template.format(label=s.label))
        cursor = s.end
    out.append(doc.text[cursor:])
    return "".join(out)
