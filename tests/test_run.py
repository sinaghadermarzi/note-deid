import json

import pytest

pytest.importorskip("faker")

from note_deid.data import Profile, build, demo_docs
from note_deid.run import SimulatedDetector, load_config, run
from note_deid.schema import Doc, read_jsonl


def _bench(tmp_path):
    sources = [Doc(f"{d.doc_id}-{i}", d.text, [], dict(d.meta)) for i in range(4) for d in demo_docs()]
    build(sources, Profile.load("configs/synphi/i2b2like.yaml"), tmp_path / "bench", seed=0, version="t")
    return tmp_path / "bench"


def test_simulated_detector_is_deterministic_and_misses_by_label(tmp_path):
    bench = _bench(tmp_path)
    docs = list(read_jsonl(bench / "train.jsonl"))
    det = SimulatedDetector("sim", miss_rate={"default": 0.0, "DATE": 1.0}, seed=3)
    a, b = det.detect(docs), det.detect(docs)
    assert a == b
    assert not any(s.label == "DATE" for spans in a for s in spans)
    assert sum(len(x) for x in a) == sum(1 for d in docs for s in d.spans if s.label != "DATE")


def test_runner_end_to_end_with_simulated_detectors(tmp_path):
    bench = _bench(tmp_path)
    cfg = load_config("configs/experiments/smoke_simulated.yaml")
    cfg["benchmark"]["path"] = str(bench)
    cfg["eval"]["bootstrap"] = {"n": 50, "seed": 0}
    out = run(cfg, out_dir=tmp_path / "results")
    assert (out / "config.yaml").exists() and (out / "summary.md").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    systems = metrics["systems"]
    assert set(systems) == {"tc", "llm", "union", "label_routed", "confidence_gated"}
    tc = systems["tc"]["strict/subtype"]["micro"]
    union = systems["union"]["strict/subtype"]["micro"]
    llm = systems["llm"]["strict/subtype"]["micro"]
    assert union["recall"] >= max(tc["recall"], llm["recall"])
    assert "by_bucket" in systems["tc"]["strict/subtype"] and "hipaa" in systems["tc"]["strict/subtype"]
    assert "vs_reference" in systems["union"]["strict/subtype"]
    assert "vs_reference" not in systems["tc"]["strict/subtype"]
    comp = json.loads((out / "complementarity.json").read_text())
    overall = comp["strict"]["overall"]
    assert overall["oracle_union_recall"] >= max(overall["recall"].values())
    assert comp["strict"]["by_group"]  # per-label breakdown present
    preds = list(read_jsonl(out / "predictions" / "union.jsonl"))
    assert len(preds) == metrics["n_docs"]
    assert json.loads((out / "cost.json").read_text()) == {}
