import os

import pytest

from note_deid.labels import OPF_TO_I2B2_CATEGORY
from note_deid.schema import Doc, Span
from note_deid.tc import (
    IGNORE_INDEX,
    chunk_windows,
    decode_tags,
    merge_window_predictions,
    regex_token_offsets,
    span_label_prob,
    spans_to_token_labels,
    tags_to_ids,
)

TEXT = "Pt John Smith (MRN 123456) seen by Dr. Lee on 03/04/2019."
SPANS = [Span(3, 13, "PATIENT"), Span(19, 25, "MEDICALRECORD"), Span(39, 42, "DOCTOR"), Span(46, 56, "DATE")]


def test_bioes_and_bio_tagging_and_round_trip():
    offs = regex_token_offsets(TEXT)
    tags = spans_to_token_labels(SPANS, offs, "bioes")
    toks = [TEXT[s:e] for s, e in offs]
    by_tok = dict(zip(toks, tags, strict=True))
    assert by_tok["John"] == "B-PATIENT" and by_tok["Smith"] == "E-PATIENT"
    assert by_tok["123456"] == "S-MEDICALRECORD" and by_tok["Lee"] == "S-DOCTOR"
    assert by_tok["03"] == "B-DATE" and by_tok["/"] == "I-DATE" and by_tok["2019"] == "E-DATE"
    decoded = decode_tags(tags, offs, text=TEXT)
    assert [(s.start, s.end, s.label) for s in decoded] == [(s.start, s.end, s.label) for s in SPANS]
    assert decoded[0].text == "John Smith"
    bio = spans_to_token_labels(SPANS, offs, "bio")
    assert dict(zip(toks, bio, strict=True))["Smith"] == "I-PATIENT"
    assert [(s.start, s.end) for s in decode_tags(bio, offs)] == [(s.start, s.end) for s in SPANS]


def test_decode_is_tolerant_and_scores_spans():
    offs = [(0, 3), (4, 8), (9, 12), (12, 12), (13, 15)]
    tags = ["I-NAME", "I-NAME", "B-DATE", "O", "E-NAME"]  # I without B; special token; dangling E
    probs = [0.9, 0.6, 0.8, 1.0, 0.7]
    spans = decode_tags(tags, offs, probs, score="min")
    assert [(s.start, s.end, s.label, s.score) for s in spans] == [
        (0, 8, "NAME", 0.6),
        (9, 12, "DATE", 0.8),
        (13, 15, "NAME", 0.7),
    ]
    mean = decode_tags(tags, offs, probs, score="mean")
    assert mean[0].score == pytest.approx(0.75)


def test_span_label_prob_sums_label_mass_over_the_extent():
    id2label = dict(enumerate(["O", "B-DATE", "E-DATE", "S-private_person"]))
    offs = [(0, 2), (3, 5), (6, 9), (9, 9)]
    dist = [[0.1, 0.5, 0.3, 0.1], [0.2, 0.1, 0.6, 0.1], [0.9, 0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0]]
    # tokens 0 and 1 overlap [0, 5): DATE mass 0.8 and 0.7 (any prefix); the special token is skipped
    assert span_label_prob(dist, offs, id2label, 0, 5, "DATE") == pytest.approx(0.75)
    assert span_label_prob(dist, offs, id2label, 0, 5, "DATE", reduce="min") == pytest.approx(0.7)
    assert span_label_prob(dist, offs, id2label, 1, 4, "DATE") == pytest.approx(0.75)  # partial overlaps count
    assert span_label_prob(dist, offs, id2label, 0, 5, "PATIENT") == 0.0  # no tag for the label
    assert span_label_prob(dist, offs, id2label, 10, 12, "DATE") == 0.0  # nothing overlaps
    # model labels are compared after the mapping (OPF private_person -> NAME category)
    assert span_label_prob(dist, offs, id2label, 6, 9, "NAME", label_map=OPF_TO_I2B2_CATEGORY) == pytest.approx(0.1)
    assert span_label_prob(dist, offs, id2label, 6, 9, "private_person") == pytest.approx(0.1)


def test_tags_to_ids_ignores_special_tokens():
    offs = [(0, 0), (0, 2), (3, 5), (0, 0)]
    ids = tags_to_ids(["O", "S-DATE", "O", "O"], offs, {"O": 0, "S-DATE": 1})
    assert ids == [IGNORE_INDEX, 1, 0, IGNORE_INDEX]


def test_chunk_windows_and_merge_prefers_central_predictions():
    assert chunk_windows(10, 20, 5) == [(0, 10)]
    assert chunk_windows(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]
    with pytest.raises(ValueError):
        chunk_windows(10, 4, 4)
    windows = [(0, 4), (2, 6)]
    per_window = [[(1, 0.9)] * 4, [(2, 0.8)] * 4]
    merged = merge_window_predictions(windows, per_window, 6)
    # tokens 2 and 3 are covered by both windows: token 2 is at the edge of window 1 (centrality 1) and at the
    # start of window 2 (centrality 0) -> window 1; token 3 is at the end of window 1 (0) vs index 1 of window 2 (1)
    assert [lab for lab, _ in merged] == [1, 1, 1, 2, 2, 2]


@pytest.mark.skipif(
    not os.environ.get("NOTE_DEID_HF_TESTS"), reason="set NOTE_DEID_HF_TESTS=1 to run model smoke tests"
)
def test_train_and_detect_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from note_deid.schema import write_jsonl
    from note_deid.tc.detector import TCDetector
    from note_deid.tc.train import train

    docs = []
    for i, name in enumerate(["John Smith", "Mary Jones", "Ahmed Khan", "Li Wei"]):
        text = f"Patient {name} was seen today. {name} is doing well."
        s = text.index(name)
        spans = [Span(s, s + len(name), "PATIENT"), Span(text.rindex(name), text.rindex(name) + len(name), "PATIENT")]
        docs.append(Doc(f"d{i}", text, spans).fill_span_text().validate())
    train_path = tmp_path / "train.jsonl"
    write_jsonl(train_path, docs)
    cfg = {
        "base_model": "google/electra-small-discriminator",
        "tokenizer": "google-bert/bert-base-uncased",  # ELECTRA shares BERT's uncased WordPiece vocabulary
        "label_space": "i2b2_28",
        "train_jsonl": str(train_path),
        "dev_jsonl": str(train_path),
        "output_dir": str(tmp_path / "model"),
        "learning_rate": 1e-4,
        "epochs": 1,
        "per_device_batch_size": 4,
        "max_length": 64,
        "precision": "fp32",
        "seeds": [1],
    }
    out = train(cfg, max_steps=300)  # ~30 s on CPU; memorizes the four training notes
    det = TCDetector(str(out), device="cpu", max_length=64, stride=8)
    preds = det.detect(docs[:1])[0]
    assert any(s.label == "PATIENT" and s.text == "John Smith" for s in preds), preds
    assert len(det.token_max_probs(docs[0].text)) > 0
    name = next(s for s in preds if s.text == "John Smith")
    assert det.forced_prob(docs[0].text, name) > 0.5
    assert det.forced_prob(docs[0].text, Span(name.start, name.end, "DATE")) < 0.5
