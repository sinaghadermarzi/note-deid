import json

import pytest

pytest.importorskip("faker")

from note_deid.data import Injector, Profile, build, demo_docs, inject_all, residual_spans, strip_placeholders
from note_deid.data.phi_injection.generators import Generators
from note_deid.data.phi_injection.templates import HEADERS, SENTENCES, render
from note_deid.eval import evaluate
from note_deid.labels import I2B2_2014_SUBTYPES
from note_deid.schema import Doc, read_jsonl

PROFILE = "configs/synphi/i2b2like.yaml"


def test_cleaning_strips_placeholders_and_labels_residuals():
    text, n = strip_placeholders("Seen by Dr. X at XYZ Hospital with (Patient name). Follow up MM/DD/YYYY.")
    assert n >= 3 and "Dr. X" not in text and "XYZ" not in text
    spans = residual_spans("A 57-year-old seen on 03/04/2019, MRN 123456, call (555) 010-2000 or a@b.org.")
    labels = {s.label: s.text for s in spans}
    assert labels["AGE"] == "57" and labels["DATE"] == "03/04/2019" and labels["MEDICALRECORD"] == "123456"
    assert labels["PHONE"] == "(555) 010-2000" and labels["EMAIL"] == "a@b.org"


def test_generators_are_deterministic_and_cover_all_subtypes():
    a, b = Generators(7).bundle(), Generators(7).bundle()
    assert a == b
    assert a.dob < a.visit < a.discharge and 0 < a.age < 105
    g = Generators(3)
    bundle = g.bundle()
    for label in (*I2B2_2014_SUBTYPES, "INSTITUTIONAL"):
        surface, labeled = g.mention(label, bundle)
        assert labeled and labeled in surface, label


def test_render_template_produces_exact_spans():
    g = Generators(5)
    bundle = g.bundle()
    text, spans = render(HEADERS[0], lambda label, form: g.mention(label, bundle, form))
    assert text.startswith("PATIENT: " + bundle.patient.full)
    for s in spans:
        assert text[s.start : s.end] == s.text
    assert {s.label for s in spans} == {"PATIENT", "MEDICALRECORD", "DATE", "DOCTOR", "HOSPITAL"}
    assert set(SENTENCES) >= set(I2B2_2014_SUBTYPES)


def test_injection_is_valid_deterministic_and_consistent():
    profile = Profile.load(PROFILE)
    profile.p_header = profile.p_signature = 1.0
    docs = demo_docs()
    out1 = inject_all(docs, profile, seed=1)
    out2 = inject_all(docs, profile, seed=1)
    assert [d.to_dict() for d in out1] == [d.to_dict() for d in out2]
    for d in out1:
        d.validate()
        assert d.meta["injected"] and d.meta["patient"] in d.text
        assert all(d.text[s.start : s.end] == s.text and s.text.strip() == s.text for s in d.spans)
        assert any(s.label == "PATIENT" for s in d.spans) and any(s.label == "DOCTOR" for s in d.spans)
        # overlapping gold spans would be a construction bug
        ordered = sorted(d.spans, key=lambda s: s.start)
        assert all(a.end <= b.start for a, b in zip(ordered, ordered[1:], strict=False))
    # source ages ("a 67-year-old") are carried into the gold standard by rule
    assert any(s.label == "AGE" and s.source == "rule" for s in out1[0].spans)
    # in-text replacement of "the patient" produces repeated patient mentions
    profile.replace_rates["PATIENT"] = 1.0
    repeated = inject_all(docs[:1], profile, seed=1)[0]
    assert sum(1 for s in repeated.spans if s.label == "PATIENT") >= 4
    assert "the patient" not in repeated.text.lower().replace(repeated.meta["patient"].lower(), "")
    different_seed = inject_all(docs, profile, seed=2)
    assert different_seed[0].meta["patient"] != out1[0].meta["patient"]


def test_tail_labels_appear_under_a_tail_stress_profile():
    profile = Profile.load(PROFILE)
    profile.weights = {k: 2.0 for k in I2B2_2014_SUBTYPES}  # every label expected ~2 per 1k tokens
    profile.target_density = 5.0
    out = inject_all(
        [Doc(f"{d.doc_id}-{i}", d.text, [], dict(d.meta)) for i in range(3) for d in demo_docs()], profile, seed=0
    )
    labels = {s.label for d in out for s in d.spans}
    assert {"FAX", "EMAIL", "URL", "DEVICE", "HEALTHPLAN", "IPADDR", "SSN", "VEHICLE", "LICENSE"} <= labels
    assert "INSTITUTIONAL" not in labels
    profile.institutional_group = True
    profile.weights["INSTITUTIONAL"] = 3.0
    out2 = inject_all(demo_docs(), profile, seed=0)
    assert any(s.label == "INSTITUTIONAL" for d in out2 for s in d.spans)


def test_build_writes_splits_and_manifest_and_gold_is_self_consistent(tmp_path):
    profile = Profile.load(PROFILE)
    sources = [Doc(f"{d.doc_id}-{i}", d.text, [], dict(d.meta)) for i in range(4) for d in demo_docs()]
    manifest = build(sources, profile, tmp_path / "synphi", seed=0, version="test")
    assert manifest["n_docs"] == 24 and set(manifest["splits"]) == {"train", "dev", "test"}
    assert sum(v["n_docs"] for v in manifest["splits"].values()) == 24
    assert all(v["n_docs"] > 0 for v in manifest["splits"].values())  # remainder carry fills small strata
    assert manifest["profile_fingerprint"] == profile.fingerprint()
    written = json.loads((tmp_path / "synphi" / "manifest.json").read_text())
    assert written["splits"]["train"]["label_counts"]
    for split in ("train", "dev", "test"):
        docs = list(read_jsonl(tmp_path / "synphi" / f"{split}.jsonl"))
        assert len(docs) == manifest["splits"][split]["n_docs"]
        for mode in ("strict", "relaxed", "relaxed_typeless"):
            rep = evaluate(docs, {d.doc_id: d.spans for d in docs}, mode=mode)
            assert rep.micro.f1 == 1.0 and rep.doc_leak_rate == 0.0
    ids = [i for s in manifest["source_ids"].values() for i in s]
    assert len(ids) == len(set(ids)) == 24


def test_injector_class_reports_targets_and_placed_counts():
    profile = Profile.load(PROFILE)
    doc = Injector(profile, seed=11).inject(demo_docs()[2])
    assert set(doc.meta["targets"]) <= set(profile.weights)
    assert sum(doc.meta["placed"].values()) == len(doc.spans)  # every placed mention is a gold span
    assert doc.meta["n_residual_rule_spans"] >= 1
