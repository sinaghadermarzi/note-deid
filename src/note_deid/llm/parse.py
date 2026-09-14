"""Tolerant parsing of LLM outputs (JSON objects, inline PHI tags, numbered segments)."""

from __future__ import annotations

import json
import re
from typing import Any

from note_deid.llm.formats import TAG_CLOSE

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)
_TAG = re.compile(r'<PHI type="([^"]+)">(.*?)' + re.escape(TAG_CLOSE), re.DOTALL)
_SEG_LINE = re.compile(r"^\s*\[(\d+)\]\s?(.*)$")


class ParseError(ValueError):
    pass


def parse_json_object(s: str) -> dict[str, Any]:
    """Extract the first JSON object from ``s`` (code fences and surrounding prose tolerated)."""
    s = _FENCE.sub("", s.strip())
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            raise ParseError("no JSON object found") from None
        try:
            obj = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ParseError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ParseError("JSON root is not an object")
    return obj


def parse_span_items(s: str) -> list[dict[str, Any]]:
    """Items of ``{"spans": [{"text", "label", ...}]}``; missing/invalid items are dropped."""
    obj = parse_json_object(s)
    items = obj.get("spans", obj.get("entities", []))
    if not isinstance(items, list):
        raise ParseError("'spans' is not a list")
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("text"), str) and it["text"] and isinstance(it.get("label"), str):
            out.append(it)
    return out


def parse_inline_tags(tagged: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Strip PHI tags; return (plain text, [(start, end, label)] over the plain text)."""
    plain: list[str] = []
    spans: list[tuple[int, int, str]] = []
    pos = 0
    out_len = 0
    for m in _TAG.finditer(tagged):
        chunk = tagged[pos : m.start()]
        plain.append(chunk)
        out_len += len(chunk)
        label, mention = m.group(1), m.group(2)
        plain.append(mention)
        if mention:
            spans.append((out_len, out_len + len(mention), label))
        out_len += len(mention)
        pos = m.end()
    plain.append(tagged[pos:])
    return "".join(plain), spans


def parse_segments(output: str) -> dict[int, str]:
    """``[n] text`` lines -> {n: text}; continuation lines are appended to the previous segment."""
    out: dict[int, str] = {}
    current: int | None = None
    for line in _FENCE.sub("", output).splitlines():
        m = _SEG_LINE.match(line)
        if m:
            current = int(m.group(1))
            out[current] = m.group(2)
        elif current is not None and line.strip():
            out[current] += "\n" + line
    return out
