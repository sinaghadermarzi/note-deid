"""i2b2/UTHealth 2014 de-identification corpus loader (gated; never redistributed).

    python -m note_deid.data.i2b2_2014 convert data/raw/i2b2_2014/testing-PHI-Gold-fixed --out data/i2b2_2014/test.jsonl

One XML file per record::

    <?xml version="1.0" encoding="UTF-8"?>
    <deIdi2b2>
    <TEXT><![CDATA[Record date: 2067-05-03 ...]]></TEXT>
    <TAGS>
    <DATE id="P0" start="13" end="23" text="2067-05-03" TYPE="DATE" comment="" />
    <NAME id="P1" start="..." end="..." text="..." TYPE="PATIENT" comment="" />
    </TAGS>
    </deIdi2b2>

The element name is the PHI category (NAME, PROFESSION, LOCATION, AGE, DATE, CONTACT, ID; some tools write ``PHI``),
``TYPE`` is the subtype, and ``start``/``end`` are character offsets into the CDATA text with an exclusive end: the same
convention as :mod:`note_deid.schema`, so no offset arithmetic is needed. Subtypes are the canonical label set
(``labels.I2B2_2014_SUBTYPES``) and are checked, not mapped. Elements of the 2014 risk-factor track (SMOKER, DIABETES,
...) that share the file format are ignored.

The corpus itself is not available in this repository (DUA; see ``docs/datasets.md``); the loader is exercised on a
synthetic fixture in ``tests/test_i2b2_2014.py``.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path

from note_deid.labels import I2B2_2014_CATEGORIES, I2B2_2014_SUBTYPES
from note_deid.schema import GOLD, Doc, Span, write_jsonl

PHI_ELEMENTS: frozenset[str] = frozenset(I2B2_2014_CATEGORIES) | {"PHI"}
SOURCE = "i2b2_2014"


def parse_i2b2_xml(xml_text: str, doc_id: str, strict: bool = True) -> Doc:
    """Parse one record. With ``strict`` a tag whose ``text`` attribute disagrees with the text slice raises
    ``ValueError`` (the corpus has a handful of such records); otherwise the slice wins and the number of such tags is
    recorded in ``meta["text_mismatches"]``."""
    root = ET.fromstring(xml_text)
    if root.tag != "deIdi2b2":
        raise ValueError(f"{doc_id}: root element is {root.tag!r}, expected 'deIdi2b2'")
    text_el = root.find("TEXT")
    if text_el is None:
        raise ValueError(f"{doc_id}: no TEXT element")
    text = text_el.text or ""
    spans: list[Span] = []
    mismatches = 0
    for el in root.iter():
        if el.tag not in PHI_ELEMENTS or el is root:
            continue
        subtype = el.get("TYPE")
        if subtype not in I2B2_2014_SUBTYPES:
            raise ValueError(f"{doc_id}: unknown PHI TYPE {subtype!r} on <{el.tag} id={el.get('id')!r}>")
        start, end = int(el.get("start", -1)), int(el.get("end", -1))
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"{doc_id}: tag {el.get('id')!r} has offsets {start}-{end} outside the text")
        slice_text = text[start:end]
        tag_text = el.get("text")
        if tag_text is not None and tag_text != slice_text:
            if strict:
                raise ValueError(
                    f"{doc_id}: tag {el.get('id')!r} text {tag_text!r} != document slice {slice_text!r} "
                    "(use strict=False to keep the slice)"
                )
            mismatches += 1
        spans.append(Span(start, end, subtype, slice_text, GOLD))
    meta = {"source": SOURCE}
    if mismatches:
        meta["text_mismatches"] = mismatches
    return Doc(doc_id, text, spans, meta).validate()


def load_i2b2_2014(path: str | Path, strict: bool = True) -> Iterator[Doc]:
    """Yield one ``Doc`` per ``*.xml`` file under ``path`` (a file or a directory, sorted by name); ``doc_id`` is the
    file stem (e.g. ``220-01``)."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.xml"))
    if not files:
        raise FileNotFoundError(f"no .xml records under {path}")
    for f in files:
        yield parse_i2b2_xml(f.read_text(encoding="utf-8"), f.stem, strict)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="convert a directory of i2b2-2014 XML records to unified JSONL")
    c.add_argument("src", help="directory of .xml records (or one file)")
    c.add_argument("--out", required=True, help="output JSONL (keep it under the gitignored data/ directory)")
    c.add_argument("--lenient", action="store_true", help="keep records whose tag text disagrees with the slice")
    ns = ap.parse_args(argv)
    docs = list(load_i2b2_2014(ns.src, strict=not ns.lenient))
    n = write_jsonl(ns.out, docs)
    counts: Counter[str] = Counter(s.label for d in docs for s in d.spans)
    print(json.dumps({"n_docs": n, "n_spans": sum(counts.values()), "label_counts": dict(sorted(counts.items()))}))


if __name__ == "__main__":
    main()
