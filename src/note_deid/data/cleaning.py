"""Source cleaning for PHI-free corpora: strip transcription placeholders and label pre-existing quasi-identifiers
(ages, dates, ...) that the source text already contains, so they are part of the gold standard."""

from __future__ import annotations

import re

from note_deid.schema import Span

PLACEHOLDERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bDr\.?\s+[A-Z]{1,3}\b\.?"), "the physician"),
    (re.compile(r"\b(?:XYZ|ABC|ABCD|XX+)\s+(?:Hospital|Medical Center|Clinic|University)\b"), "the hospital"),
    (re.compile(r"\((?:Patient|Doctor|Physician) name\)", re.IGNORECASE), "the patient"),
    (re.compile(r"\b(?:XYZ|ABCD?|XX+|YYYY|MM/DD/YYYY|DD/MM/YYYY)\b"), ""),
    (re.compile(r"\[(?:DATE|NAME|AGE|PLACE|REDACTED)\]", re.IGNORECASE), ""),
]

RESIDUAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AGE", re.compile(r"\b(\d{1,3})(?=[- ]?(?:year|yr)s?[- ]old\b)", re.IGNORECASE)),
    ("DATE", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")),
    (
        "DATE",
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b"
        ),
    ),
    ("PHONE", re.compile(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MEDICALRECORD", re.compile(r"\bMRN[:# ]*\s*(\d{5,})\b")),
]


def strip_placeholders(text: str) -> tuple[str, int]:
    """Replace transcription placeholders ("Dr. X", "XYZ Hospital", "(Patient name)") with neutral phrases."""
    n = 0
    for pat, repl in PLACEHOLDERS:
        text, k = pat.subn(repl, text)
        n += k
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text, n


def residual_spans(text: str) -> list[Span]:
    """Pre-existing quasi-identifiers found by rule (ages, dates, phones, ...), labeled with ``source='rule'``."""
    found: dict[tuple[int, int], Span] = {}
    for label, pat in RESIDUAL_PATTERNS:
        for m in pat.finditer(text):
            s, e = (m.start(1), m.end(1)) if m.groups() else (m.start(), m.end())
            if e > s and (s, e) not in found and not any(a < e and s < b for a, b in found):
                found[(s, e)] = Span(s, e, label, text[s:e], "rule")
    return sorted(found.values(), key=lambda x: (x.start, x.end))
