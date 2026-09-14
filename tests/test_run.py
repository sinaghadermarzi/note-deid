import json

import pytest

pytest.importorskip("faker")

from note_deid.data import Profile, demo_docs
from note_deid.data.synphi import build
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


def test_confidence_gate_uses_the_detectors_verifier_and_forced_probability(tmp_path):
    bench = _bench(tmp_path)
    cfg = load_config("configs/experiments/smoke_simulated.yaml")
    cfg["benchmark"]["path"] = str(bench)
    cfg["eval"]["bootstrap"] = {"n": 20, "seed": 0}
    # theta_hi above the simulated TC score (0.95): every TC-only span goes to the verifier; LLM false positives on
    # frequent labels get a low forced probability and go to the verifier too
    cfg["fusion"] = [{"name": "confidence_gated", "theta_hi": 0.99, "theta_lo": 0.3, "tau": 10}]
    out = run(cfg, out_dir=tmp_path / "with_hooks")
    stats = json.loads((out / "metrics.json").read_text())["fusion"]["confidence_gated"]
    assert stats["verifier_calls"] > 0
    assert not any(rule.endswith(":no-verifier") for rule in stats["rules"])
    assert "tc-only:low:verified" in stats["rules"]
    rows = [
        json.loads(line) for line in (out / "predictions" / "confidence_gated.decisions.jsonl").read_text().splitlines()
    ]
    assert rows and {"doc_id", "start", "end", "label", "rule", "accepted"} <= set(rows[0])
    assert sum(1 for r in rows if r["rule"].endswith(":verified")) == stats["verifier_calls"]
    assert "verifier calls" in (out / "summary.md").read_text()
    # the oracle verifier keeps every gold-overlapping candidate, so the gate cannot lose recall to the TC alone
    systems = json.loads((out / "metrics.json").read_text())["systems"]
    gate = systems["confidence_gated"]["relaxed/subtype"]["micro"]
    assert gate["recall"] >= systems["tc"]["relaxed/subtype"]["micro"]["recall"]

    cfg["fusion"] = [{"name": "confidence_gated", "theta_hi": 0.99, "verify": False}]
    out2 = run(cfg, out_dir=tmp_path / "without_verifier")
    stats2 = json.loads((out2 / "metrics.json").read_text())["fusion"]["confidence_gated"]
    assert stats2["verifier_calls"] == 0
    assert any(rule.endswith(":no-verifier") for rule in stats2["rules"])
