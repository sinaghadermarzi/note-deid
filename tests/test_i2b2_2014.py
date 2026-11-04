"""The i2b2-2014 loader on a synthetic record in the corpus' XML format (the corpus itself is gated)."""

import json

import pytest

from note_deid.data.i2b2_2014 import load_i2b2_2014, main, parse_i2b2_xml
from note_deid.schema import read_jsonl

TEXT = (
    "Record date: 2067-05-03\n\n"
    "Mercy Hospital\n"
    "Patient: Jane Doe, 92 y/o retired teacher, MRN 4521980.\n"
    "Seen by Dr. Lee. Call 555-0100 with questions. Lives in Boston.\n"
)


def _offsets(fragment: str) -> tuple[int, int]:
    start = TEXT.index(fragment)
    return start, start + len(fragment)


def _tag(element: str, idx: int, fragment: str, subtype: str, text: str | None = None) -> str:
    start, end = _offsets(fragment)
    shown = fragment if text is None else text
    return f'<{element} id="P{idx}" start="{start}" end="{end}" text="{shown}" TYPE="{subtype}" comment="" />'


def _record(tags: list[str], root: str = "deIdi2b2") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<{root}>\n<TEXT><![CDATA[{TEXT}]]></TEXT>\n<TAGS>\n" + "\n".join(tags) + f"\n</TAGS>\n</{root}>\n"
    )


GOLD_TAGS = [
    _tag("DATE", 0, "2067-05-03", "DATE"),
    _tag("LOCATION", 1, "Mercy Hospital", "HOSPITAL"),
    _tag("NAME", 2, "Jane Doe", "PATIENT"),
    _tag("AGE", 3, "92", "AGE"),
    _tag("PROFESSION", 4, "retired teacher", "PROFESSION"),
    _tag("ID", 5, "4521980", "MEDICALRECORD"),
    _tag("PHI", 6, "Lee", "DOCTOR"),  # some tools emit the generic PHI element
    _tag("CONTACT", 7, "555-0100", "PHONE"),
    _tag("LOCATION", 8, "Boston", "CITY"),
    # a risk-factor-track element sharing the file format is ignored
    '<SMOKER id="S0" start="-1" end="-1" status="unknown" />',
]


def test_parse_record_preserves_cdata_text_and_offsets():
    doc = parse_i2b2_xml(_record(GOLD_TAGS), "220-01")
    assert doc.text == TEXT and doc.doc_id == "220-01" and doc.meta == {"source": "i2b2_2014"}
    assert [(s.label, s.text) for s in doc.sorted_spans()] == [
        ("DATE", "2067-05-03"),
        ("HOSPITAL", "Mercy Hospital"),
        ("PATIENT", "Jane Doe"),
        ("AGE", "92"),
        ("PROFESSION", "retired teacher"),
        ("MEDICALRECORD", "4521980"),
        ("DOCTOR", "Lee"),
        ("PHONE", "555-0100"),
        ("CITY", "Boston"),
    ]
    assert all(s.source == "gold" for s in doc.spans)


def test_text_mismatch_is_an_error_unless_lenient():
    tags = [_tag("NAME", 0, "Jane Doe", "PATIENT", text="Jane Do")]
    with pytest.raises(ValueError, match="document slice"):
        parse_i2b2_xml(_record(tags), "x")
    doc = parse_i2b2_xml(_record(tags), "x", strict=False)
    assert doc.spans[0].text == "Jane Doe" and doc.meta["text_mismatches"] == 1


@pytest.mark.parametrize(
    "tags, message",
    [
        ([_tag("NAME", 0, "Jane Doe", "NICKNAME")], "unknown PHI TYPE"),
        (['<DATE id="P0" start="5" end="5000" text="x" TYPE="DATE" />'], "outside the text"),
        (['<DATE id="P0" start="7" end="3" text="x" TYPE="DATE" />'], "outside the text"),
    ],
)
def test_bad_tags_are_rejected(tags, message):
    with pytest.raises(ValueError, match=message):
        parse_i2b2_xml(_record(tags), "x")


def test_wrong_root_is_rejected():
    with pytest.raises(ValueError, match="deIdi2b2"):
        parse_i2b2_xml(_record(GOLD_TAGS, root="other"), "x")


def test_directory_loader_and_convert_cli(tmp_path, capsys):
    src = tmp_path / "gold"
    src.mkdir()
    (src / "220-02.xml").write_text(_record(GOLD_TAGS[:3]), encoding="utf-8")
    (src / "220-01.xml").write_text(_record(GOLD_TAGS), encoding="utf-8")
    (src / "notes.txt").write_text("ignored", encoding="utf-8")
    docs = list(load_i2b2_2014(src))
    assert [d.doc_id for d in docs] == ["220-01", "220-02"]  # sorted by file name
    assert len(docs[1].spans) == 3
    with pytest.raises(FileNotFoundError):
        list(load_i2b2_2014(tmp_path / "empty"))

    out = tmp_path / "data" / "test.jsonl"
    main(["convert", str(src), "--out", str(out)])
    report = json.loads(capsys.readouterr().out)
    assert report["n_docs"] == 2 and report["n_spans"] == 12
    assert report["label_counts"]["DATE"] == 2 and report["label_counts"]["CITY"] == 1
    assert [d.doc_id for d in read_jsonl(out)] == ["220-01", "220-02"]
