import csv
import json

import pytest

from note_deid.summarize import PAIR_COLUMNS, SYSTEM_COLUMNS, main, pair_rows, summarize_run, system_rows


def _micro(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn, "precision": p, "recall": r, "f1": 0.5}


def _entry(tp, fp, fn, **extra):
    return {"n_docs": 3, "micro": _micro(tp, fp, fn), "macro": {"f1": 0.4}, "doc_leak_rate": 0.25, **extra}


def test_system_rows_flatten_views_and_reference_deltas():
    metrics = {
        "systems": {
            "tc": {"strict/subtype": _entry(8, 1, 2, hipaa=_entry(5, 0, 1)), "strict/category": _entry(9, 1, 1)},
            "union": {
                "strict/subtype": _entry(
                    10,
                    2,
                    0,
                    vs_reference={"recall": {"delta": 0.2, "ci_low": 0.1, "ci_high": 0.3, "p_value": 0.01}},
                )
            },
        }
    }
    rows = system_rows(metrics, "run-1", reference="tc")
    assert [(r["system"], r["mode"], r["level"], r["view"]) for r in rows] == [
        ("tc", "strict", "subtype", "full"),
        ("tc", "strict", "subtype", "hipaa"),
        ("tc", "strict", "category", "full"),
        ("union", "strict", "subtype", "full"),
    ]
    assert rows[1]["support"] == 6 and rows[1]["macro_f1"] == 0.4 and rows[0]["run_id"] == "run-1"
    assert rows[3]["reference"] == "tc" and rows[3]["delta_recall"] == 0.2 and rows[3]["delta_recall_p"] == 0.01
    assert "reference" not in rows[0]


def test_pair_rows_pull_recalls_from_the_overall_block():
    comp = {
        "strict": {
            "overall": {
                "n_gold": 23,
                "recall": {"tc": 0.78, "llm": 0.8},
                "oracle_union_recall": 0.91,
                "intersection_recall": 0.65,
                "pairs": [{"a": "tc", "b": "llm", "fn_a": 5, "fn_b": 5, "fn_both": 2, "fn_either": 8, "jaccard": 0.25}],
            }
        }
    }
    (row,) = pair_rows(comp, "run-1")
    assert row["recall_a"] == 0.78 and row["recall_b"] == 0.8 and row["fn_both"] == 2 and row["mode"] == "strict"
    assert "kappa" not in row  # absent fields stay absent; the CSV writer blanks them


def test_summarize_cli_on_a_simulated_run(tmp_path, capsys):
    pytest.importorskip("faker")
    from note_deid.data import Profile, demo_docs
    from note_deid.data.synphi import build
    from note_deid.run import load_config, run
    from note_deid.schema import Doc

    sources = [Doc(f"{d.doc_id}-{i}", d.text, [], dict(d.meta)) for i in range(3) for d in demo_docs()]
    build(sources, Profile.load("configs/synphi/i2b2like.yaml"), tmp_path / "bench", seed=0, version="t")
    cfg = load_config("configs/experiments/smoke_simulated.yaml")
    cfg["benchmark"]["path"] = str(tmp_path / "bench")
    cfg["eval"]["bootstrap"] = {"n": 20, "seed": 0}
    run_dir = run(cfg, out_dir=tmp_path / "results")

    main([str(run_dir), "--out", str(tmp_path / "summary" / "smoke")])
    printed = capsys.readouterr().out.splitlines()
    assert [p.rsplit("/", 1)[1] for p in printed] == ["smoke.csv", "smoke.complementarity.csv"]

    with (tmp_path / "summary" / "smoke.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(SYSTEM_COLUMNS)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    n_full = sum(len(v) for v in metrics["systems"].values())
    n_hipaa = sum(1 for v in metrics["systems"].values() for e in v.values() if e.get("hipaa"))
    assert len(rows) == n_full + n_hipaa
    union = next(r for r in rows if r["system"] == "union" and r["mode"] == "strict" and r["level"] == "subtype")
    assert union["view"] == "full" and union["reference"] == "tc" and union["delta_recall"] != ""
    assert float(union["recall"]) == metrics["systems"]["union"]["strict/subtype"]["micro"]["recall"]
    tc = next(r for r in rows if r["system"] == "tc" and r["mode"] == "strict" and r["level"] == "subtype")
    assert tc["reference"] == "" and tc["run_id"] == run_dir.name

    with (tmp_path / "summary" / "smoke.complementarity.csv").open(newline="") as fh:
        pairs = list(csv.DictReader(fh))
    assert list(pairs[0]) == list(PAIR_COLUMNS) and {p["mode"] for p in pairs} == {"strict", "relaxed"}
    assert float(pairs[0]["oracle_union_recall"]) >= max(float(pairs[0]["recall_a"]), float(pairs[0]["recall_b"]))
    assert summarize_run(run_dir, tmp_path / "again" / "x")[0].name == "x.csv"
