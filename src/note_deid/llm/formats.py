"""Prompt construction for the LLM detector: output formats F1-F4 (framework-designs.md H15, Appendix B), the
TC-first editor prompt (H5) and the span verifier prompt (H3)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from note_deid.labels import I2B2_2014_SUBTYPES
from note_deid.schema import Span

FORMATS: tuple[str, ...] = ("json_strings", "json_anchored", "inline_tags", "segments")
TAG_OPEN = '<PHI type="{label}">'
TAG_CLOSE = "</PHI>"

DEFAULT_GUIDELINE = """You are a clinical de-identification annotator. Find every mention of protected health
information (PHI) and quasi-identifiers in the clinical note, following the i2b2-2014 scheme. Labels and definitions:
NAME: PATIENT (patient and relatives' names), DOCTOR (clinician names), USERNAME (login names);
PROFESSION: the patient's occupation or employer-identifying job title;
LOCATION: HOSPITAL (hospital/clinic names), ORGANIZATION (employers, other organizations), STREET, CITY, STATE,
COUNTRY, ZIP, LOCATION-OTHER (landmarks, buildings);
AGE: any stated age; DATE: any date or partial date (months, days, years, holidays; not durations);
CONTACT: PHONE, FAX, EMAIL, URL, IPADDR;
ID: SSN, MEDICALRECORD, HEALTHPLAN, ACCOUNT, LICENSE, VEHICLE, DEVICE, BIOID, IDNUM (any other identifier).
Annotate every occurrence, including repeated mentions, headers, footers and signature blocks. Do not annotate
generic role words (the patient, the doctor), drug names, diseases, or units. Use the exact text as it appears."""


def load_guideline(path: str | Path | None = None, text: str | None = None) -> str:
    if text is not None:
        return text
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_GUIDELINE


def _label_line(labels: Sequence[str]) -> str:
    return "Allowed labels: " + ", ".join(labels) + "."


def _instructions(fmt: str, labels: Sequence[str]) -> str:
    if fmt == "json_strings":
        return (
            _label_line(labels)
            + '\nReturn only a JSON object: {"spans": [{"text": "<exact text>", "label": "<LABEL>"}, ...]}.'
            " List one item per distinct PHI string; every occurrence of that string will be annotated."
        )
    if fmt == "json_anchored":
        return (
            _label_line(labels) + '\nReturn only a JSON object: {"spans": [{"text": "<exact text>", "label": "<LABEL>",'
            ' "before": "<the 1-4 words immediately preceding this occurrence>"}, ...]}.'
            " List one item per occurrence, in document order, with the exact preceding words as the anchor."
        )
    if fmt in ("inline_tags", "editor"):
        return (
            _label_line(labels)
            + "\nReturn the complete input text verbatim, unchanged except that every PHI mention is"
            f" wrapped as {TAG_OPEN.format(label='LABEL')}mention{TAG_CLOSE}. Do not add, remove, reorder or rewrite"
            " any other characters. Do not use markdown fences."
        )
    if fmt == "segments":
        return (
            _label_line(labels) + "\nThe input is a list of numbered segments. Return the same segments, one per line,"
            f" each starting with its number in brackets, with every PHI mention wrapped as"
            f" {TAG_OPEN.format(label='LABEL')}mention{TAG_CLOSE}. Keep every other character unchanged."
        )
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")


def build_messages(
    fmt: str, text: str, guideline: str | None = None, labels: Sequence[str] = I2B2_2014_SUBTYPES
) -> list[dict[str, str]]:
    """Messages for a from-scratch extraction in format ``fmt``."""
    system = load_guideline(text=guideline) + "\n\n" + _instructions(fmt, labels)
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


# -- segments (F4) -------------------------------------------------------------------------------------------------
_PARA = re.compile(r"\n\s*\n")


def segment_text(text: str, max_chars: int = 1500) -> list[tuple[int, int]]:
    """Paragraph-based segments (start, end) of at most ``max_chars`` (long paragraphs split at line breaks/spaces)."""
    segs: list[tuple[int, int]] = []
    pos = 0
    boundaries = [m.end() for m in _PARA.finditer(text)] + [len(text)]
    for b in boundaries:
        start, end = pos, b
        pos = b
        while end - start > max_chars:
            cut = text.rfind("\n", start, start + max_chars)
            if cut <= start:
                cut = text.rfind(" ", start, start + max_chars)
            if cut <= start:
                cut = start + max_chars
            segs.append((start, cut))
            start = cut
        if end > start:
            segs.append((start, end))
    return segs


def render_segments(text: str, segs: Sequence[tuple[int, int]]) -> str:
    return "\n".join(f"[{i + 1}] {text[s:e].replace(chr(10), ' ')}" for i, (s, e) in enumerate(segs))


def build_segment_messages(
    text: str,
    segs: Sequence[tuple[int, int]],
    guideline: str | None = None,
    labels: Sequence[str] = I2B2_2014_SUBTYPES,
) -> list[dict[str, str]]:
    system = load_guideline(text=guideline) + "\n\n" + _instructions("segments", labels)
    return [{"role": "system", "content": system}, {"role": "user", "content": render_segments(text, segs)}]


# -- editor (H5) ---------------------------------------------------------------------------------------------------
def render_tagged(text: str, spans: Sequence[Span]) -> str:
    """Insert PHI tags around ``spans`` (non-overlapping, any order)."""
    out: list[str] = []
    cursor = 0
    for s in sorted(spans, key=lambda x: (x.start, x.end)):
        if s.start < cursor:
            continue
        out.append(text[cursor : s.start])
        out.append(TAG_OPEN.format(label=s.label) + text[s.start : s.end] + TAG_CLOSE)
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)


def build_editor_messages(
    text: str,
    pretagged: Sequence[Span],
    guideline: str | None = None,
    labels: Sequence[str] = I2B2_2014_SUBTYPES,
    focus_labels: Sequence[str] = (),
) -> list[dict[str, str]]:
    """H5: the LLM reviews a note pre-tagged by the token classifier and returns the corrected tagged text."""
    extra = ""
    if focus_labels:
        extra = (
            "\nThe pre-tagging system cannot produce these labels, so check for them with particular care: "
            + ", ".join(focus_labels)
        )
    system = (
        load_guideline(text=guideline)
        + "\n\nThe input below was pre-tagged by an automatic system. Review it: add tags for PHI it missed,"
        " remove tags that are not PHI, and correct wrong labels or boundaries.\n"
        + _instructions("editor", labels)
        + extra
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": render_tagged(text, pretagged)}]


# -- verifier (H3) -------------------------------------------------------------------------------------------------
def build_verifier_messages(
    text: str, span: Span, guideline: str | None = None, context_chars: int = 200
) -> list[dict[str, str]]:
    left = text[max(0, span.start - context_chars) : span.start]
    right = text[span.end : span.end + context_chars]
    mention = text[span.start : span.end]
    system = (
        load_guideline(text=guideline) + "\n\nDecide whether the candidate mention is PHI under the guideline."
        ' Return only JSON: {"is_phi": true|false, "label": "<LABEL or null>"}.'
    )
    user = f'Context:\n...{left}[[{mention}]]{right}...\n\nCandidate: "{mention}" proposed label: {span.label}'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
