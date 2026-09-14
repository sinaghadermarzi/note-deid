"""LLM backend: LiteLLM client (aliases, cache, cost log, concurrency 1), prompt formats F1-F4 plus editor (H5) and
verifier (H3), tolerant parsers, offset aligners, and ``LLMDetector``. ``litellm`` is imported lazily inside the
client so the rest of the package works without it.
"""

from note_deid.llm.align import AlignStats, align_items, align_segments, align_tagged
from note_deid.llm.client import LLMCall, LLMClient, LLMResponse
from note_deid.llm.detector import DetectResult, LLMDetector
from note_deid.llm.formats import (
    FORMATS,
    build_editor_messages,
    build_messages,
    build_segment_messages,
    build_verifier_messages,
    render_tagged,
    segment_text,
)
from note_deid.llm.parse import ParseError, parse_inline_tags, parse_json_object, parse_segments, parse_span_items

__all__ = [
    "FORMATS",
    "AlignStats",
    "DetectResult",
    "LLMCall",
    "LLMClient",
    "LLMDetector",
    "LLMResponse",
    "ParseError",
    "align_items",
    "align_segments",
    "align_tagged",
    "build_editor_messages",
    "build_messages",
    "build_segment_messages",
    "build_verifier_messages",
    "parse_inline_tags",
    "parse_json_object",
    "parse_segments",
    "parse_span_items",
    "render_tagged",
    "segment_text",
]
