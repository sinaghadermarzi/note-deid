import pytest

from note_deid.labels import I2B2_TO_OPF
from note_deid.schema import Doc, Span, from_opf_record, read_jsonl, redact, to_opf_record, write_jsonl

TEXT = "Pt John Smith (MRN 123456) seen by Dr. Lee on 03/04/2019 at Mercy Hospital. Occupation: welder."


def make_doc() -> Doc:
    spans = [
        Span(3, 13, "PATIENT"),
        Span(19, 25, "MEDICALRECORD"),
        Span(39, 42, "DOCTOR"),
        Span(46, 56, "DATE"),
        Span(60, 74, "HOSPITAL"),
        Span(88, 94, "PROFESSION"),
    ]
    return Doc("d1", TEXT, spans, {"source": "unit-test"}).fill_span_text().validate()


def test_span_text_matches_document():
    doc = make_doc()
    expected = ["John Smith", "123456", "Lee", "03/04/2019", "Mercy Hospital", "welder"]
    assert [s.text for s in doc.sorted_spans()] == expected


def test_invalid_spans_are_rejected():
    with pytest.raises(ValueError):
        Span(5, 5, "DATE")
    with pytest.raises(ValueError):
        Doc("bad", "short", [Span(0, 99, "DATE")]).validate()
    with pytest.raises(ValueError):
        Doc("bad", "short text", [Span(0, 5, "DATE", text="wrong")]).validate()


def test_jsonl_round_trip(tmp_path):
    docs = [make_doc(), Doc("d2", "no phi here", [], {})]
    path = tmp_path / "docs.jsonl"
    assert write_jsonl(path, docs) == 2
    back = list(read_jsonl(path))
    assert [d.to_dict() for d in back] == [d.to_dict() for d in docs]


def test_opf_record_conversion_drops_unsupported_labels():
    doc = make_doc()
    rec = to_opf_record(doc, label_map=I2B2_TO_OPF)
    assert rec["text"] == TEXT
    assert rec["spans"]["private_person: John Smith"] == [[3, 13]]
    assert rec["spans"]["account_number: 123456"] == [[19, 25]]
    assert rec["spans"]["private_date: 03/04/2019"] == [[46, 56]]
    # HOSPITAL and PROFESSION have no OPF label and must be dropped
    assert not any(k.endswith("Mercy Hospital") or k.endswith("welder") for k in rec["spans"])
    back = from_opf_record(rec)
    assert back.doc_id == "d1"
    assert {(s.start, s.end, s.label) for s in back.spans} == {
        (3, 13, "private_person"),
        (19, 25, "account_number"),
        (39, 42, "private_person"),
        (46, 56, "private_date"),
    }


def test_redact_merges_overlaps():
    doc = Doc("d", "call 555-0100 now", [Span(5, 13, "PHONE"), Span(9, 13, "PHONE")])
    assert redact(doc) == "call [PHONE] now"
