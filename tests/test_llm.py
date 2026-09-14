import json

import pytest

from note_deid.llm import (
    AlignStats,
    LLMClient,
    LLMDetector,
    align_items,
    align_segments,
    align_tagged,
    build_editor_messages,
    parse_inline_tags,
    parse_json_object,
    parse_segments,
    render_tagged,
    segment_text,
)
from note_deid.schema import Doc, Span

TEXT = "Pt John Smith (MRN 123456) seen by Dr. Lee on 03/04/2019.\nJohn Smith will return in 2 weeks."
CONFIG = {
    "model_list": [{"model_name": "llm-test", "litellm_params": {"model": "openai/gpt-5", "temperature": 0}}],
    "litellm_settings": {"cache": True},
    "note_deid": {"concurrency": 1},
}


# -- parsing ---------------------------------------------------------------------------------------------------------
def test_parse_json_tolerates_fences_and_prose():
    raw = 'Sure:\n```json\n{"spans": [{"text": "John Smith", "label": "PATIENT"}]}\n```'
    assert parse_json_object(raw)["spans"][0]["label"] == "PATIENT"


def test_parse_inline_tags_and_segments():
    plain, spans = parse_inline_tags('Pt <PHI type="PATIENT">John Smith</PHI> seen <PHI type="DATE">today</PHI>.')
    assert plain == "Pt John Smith seen today."
    assert spans == [(3, 13, "PATIENT"), (19, 24, "DATE")]
    segs = parse_segments("[1] first line\n[2] second\ncontinued\n[3] third")
    assert segs == {1: "first line", 2: "second\ncontinued", 3: "third"}


# -- alignment -------------------------------------------------------------------------------------------------------
def test_align_items_exact_all_occurrences_whitespace_and_fuzzy():
    stats = AlignStats()
    items = [
        {"text": "John Smith", "label": "PATIENT"},  # two occurrences
        {"text": "Dr.  Lee", "label": "DOCTOR"},  # whitespace drift
        {"text": "03/04/2O19", "label": "DATE"},  # OCR-like typo -> fuzzy
        {"text": "Mercy Hospital", "label": "HOSPITAL"},  # absent
    ]
    spans = align_items(TEXT, items, occurrences="all", stats=stats)
    found = {(s.start, s.end, s.label) for s in spans}
    assert (3, 13, "PATIENT") in found and (58, 68, "PATIENT") in found
    assert any(s.label == "DOCTOR" and s.text == "Dr. Lee" for s in spans)
    assert any(s.label == "DATE" and s.text == "03/04/2019" and s.score < 1 for s in spans)
    assert stats.exact == 1 and stats.whitespace == 1 and stats.fuzzy == 1 and stats.failed == 1
    first_only = align_items(TEXT, items[:1], occurrences="first")
    assert [(s.start, s.end) for s in first_only] == [(3, 13)]


def test_align_items_anchored_picks_the_right_occurrence():
    items = [{"text": "John Smith", "label": "PATIENT", "before": "2019."}]
    spans = align_items(TEXT, items)
    assert [(s.start, s.end) for s in spans] == [(58, 68)]


def test_align_tagged_accepts_verbatim_and_whitespace_only_changes_and_rejects_rewrites():
    tagged = TEXT.replace("John Smith", '<PHI type="PATIENT">John Smith</PHI>', 1)
    spans, ok = align_tagged(TEXT, tagged)
    assert ok and [(s.start, s.end, s.label) for s in spans] == [(3, 13, "PATIENT")]
    ws = tagged.replace("\n", "  \n ")  # extra whitespace only
    spans_ws, ok_ws = align_tagged(TEXT, ws)
    assert ok_ws and [(s.start, s.end) for s in spans_ws] == [(3, 13)]
    rewritten = tagged.replace("will return", "returns")
    assert align_tagged(TEXT, rewritten) == ([], False)


def test_segments_round_trip():
    long_text = "\n\n".join(f"Paragraph {i} mentions Dr. Lee here." for i in range(6))
    segs = segment_text(long_text, max_chars=60)
    assert segs[0][0] == 0 and segs[-1][1] == len(long_text)
    outputs = {
        i + 1: long_text[s:e].replace("Dr. Lee", '<PHI type="DOCTOR">Dr. Lee</PHI>') for i, (s, e) in enumerate(segs)
    }
    spans = align_segments(long_text, segs, outputs)
    assert len(spans) == 6 and all(s.text == "Dr. Lee" for s in spans)


def test_render_tagged_and_editor_prompt():
    spans = [Span(3, 13, "PATIENT"), Span(46, 56, "DATE")]
    tagged = render_tagged(TEXT, spans)
    assert tagged.startswith('Pt <PHI type="PATIENT">John Smith</PHI>')
    msgs = build_editor_messages(TEXT, spans, focus_labels=["AGE", "PROFESSION"])
    assert "pre-tagged" in msgs[0]["content"] and "AGE, PROFESSION" in msgs[0]["content"]
    assert msgs[1]["content"] == tagged


# -- client + detector with mocked completions --------------------------------------------------------------------
def test_client_caches_and_logs(tmp_path):
    pytest.importorskip("litellm")
    client = LLMClient(CONFIG, cache_dir=tmp_path / "cache")
    msgs = [{"role": "user", "content": "hello"}]
    r1 = client.complete("llm-test", msgs, mock_response="hi there")
    r2 = client.complete("llm-test", msgs, mock_response="hi there")
    assert r1.text == "hi there" and r2.text == "hi there"
    assert r1.call.cached is False and r2.call.cached is True
    summary = client.cost_summary()
    assert summary["calls"] == 2 and summary["cached_calls"] == 1
    log = client.write_log(tmp_path / "calls.jsonl")
    assert len(log.read_text().splitlines()) == 2
    with pytest.raises(KeyError):
        client.resolve("nope")


def test_detector_formats_with_mocked_llm(tmp_path):
    pytest.importorskip("litellm")
    client = LLMClient(CONFIG, cache_dir=tmp_path / "cache")
    doc = Doc("d1", TEXT)
    json_out = json.dumps(
        {"spans": [{"text": "John Smith", "label": "PATIENT"}, {"text": "123456", "label": "MEDICALRECORD"}]}
    )
    det = LLMDetector(client, "llm-test", fmt="json_strings", mock_response=json_out)
    spans = det.detect([doc])[0]
    assert {(s.start, s.end, s.label) for s in spans} == {
        (3, 13, "PATIENT"),
        (58, 68, "PATIENT"),
        (19, 25, "MEDICALRECORD"),
    }
    assert det.results["d1"].stats.exact == 2
    bad = LLMDetector(client, "llm-test", fmt="json_strings", mock_response="not json at all")
    assert bad.detect([doc]) == [[]] and bad.results["d1"].parse_failed
    tagged_out = TEXT.replace("03/04/2019", '<PHI type="DATE">03/04/2019</PHI>')
    inline = LLMDetector(client, "llm-test", fmt="inline_tags", mock_response=tagged_out)
    assert [(s.start, s.end, s.label) for s in inline.detect([doc])[0]] == [(46, 56, "DATE")]
    edited = inline.edit(doc, [Span(3, 13, "PATIENT")])
    assert any(s.label == "DATE" for s in edited.spans)
    verifier = LLMDetector(client, "llm-test", fmt="json_strings", mock_response='{"is_phi": true, "label": "DATE"}')
    assert verifier.verify(doc, Span(46, 56, "DATE")) is True
    with pytest.raises(ValueError):
        LLMDetector(client, "llm-test", fmt="xml")
