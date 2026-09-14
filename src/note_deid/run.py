"""Config-driven experiment runner.

    python -m note_deid.run --config configs/experiments/smoke_simulated.yaml [--limit N] [--out DIR]

Pipeline: load a benchmark split -> run detectors (tc / llm / saved predictions / simulated) -> apply fusion
policies -> evaluate every prediction set (modes x levels, HIPAA view, frequency buckets, paired bootstrap against a
reference system) -> complementarity statistics across base detectors -> write ``results/<name>/<run-id>/``
containing config.yaml, predictions/*.jsonl, metrics.json, complementarity.json, cost.json and summary.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from note_deid.eval import (
    by_bucket,
    complementarity,
    evaluate,
    evaluate_docs,
    label_support,
    paired_bootstrap,
    report_markdown,
)
from note_deid.hybrid import GateConfig, confidence_gated, label_routed, union
from note_deid.labels import OPENMED_TO_I2B2, OPF_TO_I2B2_CATEGORY
from note_deid.schema import Doc, Span, read_jsonl, write_jsonl

LABEL_MAPS = {"opf_category": OPF_TO_I2B2_CATEGORY, "openmed": OPENMED_TO_I2B2, None: None, "none": None}


# -- detectors -----------------------------------------------------------------------------------------------------
class SimulatedDetector:
    """Gold spans with deterministic misses and false positives (pipeline tests; system simulations).

    ``miss_rate`` is global or per label ({"default": 0.1, "EMAIL": 0.9}); ``fp_rate`` adds spurious spans per gold
    span; ``boundary_noise`` shrinks a fraction of spans by one character (strict vs relaxed differences).
    """

    def __init__(
        self,
        name: str,
        miss_rate: float | Mapping[str, float] = 0.1,
        fp_rate: float = 0.0,
        boundary_noise: float = 0.0,
        score: float = 0.9,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.miss_rate = miss_rate
        self.fp_rate = fp_rate
        self.boundary_noise = boundary_noise
        self.score = score
        self.seed = seed

    def _rate(self, label: str) -> float:
        if isinstance(self.miss_rate, Mapping):
            return float(self.miss_rate.get(label, self.miss_rate.get("default", 0.0)))
        return float(self.miss_rate)

    def detect(self, docs: Sequence[Doc]) -> list[list[Span]]:
        out: list[list[Span]] = []
        for d in docs:
            rng = random.Random(f"{self.seed}:{self.name}:{d.doc_id}")
            spans: list[Span] = []
            for s in d.spans:
                if rng.random() < self._rate(s.label):
                    continue
                start, end = s.start, s.end
                if self.boundary_noise and rng.random() < self.boundary_noise and end - start > 2:
                    end -= 1
                spans.append(Span(start, end, s.label, d.text[start:end], self.name, self.score))
                if self.fp_rate and rng.random() < self.fp_rate:
                    pos = rng.randrange(0, max(1, len(d.text) - 6))
                    fp_end = min(len(d.text), pos + 5)
                    if not any(x.start < fp_end and pos < x.end for x in d.spans) and fp_end > pos:
                        spans.append(Span(pos, fp_end, s.label, d.text[pos:fp_end], self.name, self.score / 2))
            out.append(spans)
        return out


class SavedPredictions:
    def __init__(self, name: str, path: str | Path) -> None:
        self.name = name
        self.by_id = {d.doc_id: d.spans for d in read_jsonl(path)}

    def detect(self, docs: Sequence[Doc]) -> list[list[Span]]:
        return [list(self.by_id.get(d.doc_id, [])) for d in docs]


def build_detector(name: str, cfg: Mapping[str, Any], llm_client: Any = None) -> Any:
    kind = cfg.get("type", name)
    if kind == "simulated":
        return SimulatedDetector(
            name,
            cfg.get("miss_rate", 0.1),
            cfg.get("fp_rate", 0.0),
            cfg.get("boundary_noise", 0.0),
            cfg.get("score", 0.9),
            cfg.get("seed", 0),
        )
    if kind == "predictions":
        return SavedPredictions(name, cfg["path"])
    if kind == "tc":
        from note_deid.tc.detector import TCDetector

        det = TCDetector(
            cfg["model"],
            label_map=LABEL_MAPS.get(cfg.get("label_map")),
            device=cfg.get("device"),
            max_length=cfg.get("max_length"),
            stride=int(cfg.get("stride", 64)),
            batch_size=int(cfg.get("batch_size", 8)),
            score=cfg.get("score", "min"),
            tokenizer=cfg.get("tokenizer"),
        )
        det.name = name
        return det
    if kind == "llm":
        from note_deid.llm.client import LLMClient
        from note_deid.llm.detector import LLMDetector

        client = llm_client or LLMClient(cfg.get("client_config", "configs/litellm.models.yaml"))
        det = LLMDetector(
            client,
            cfg["alias"],
            fmt=cfg.get("format", "json_strings"),
            guideline_path=cfg.get("guideline"),
            occurrences=cfg.get("occurrences", "all"),
            fuzzy_threshold=float(cfg.get("fuzzy_threshold", 90.0)),
            segment_chars=int(cfg.get("segment_chars", 1500)),
            max_tokens=cfg.get("max_tokens"),
            mock_response=cfg.get("mock_response"),
        )
        det.name = name
        return det
    raise ValueError(f"unknown detector type {kind!r} for {name!r}")


# -- fusion ----------------------------------------------------------------------------------------------------------
def apply_fusion(
    spec: Mapping[str, Any], docs: Sequence[Doc], preds: Mapping[str, list[list[Span]]], train_counts: Mapping[str, int]
) -> list[list[Span]]:
    kind = spec["name"]
    tc, llm = spec.get("tc", "tc"), spec.get("llm", "llm")
    out: list[list[Span]] = []
    for i, d in enumerate(docs):
        outputs = {k: v[i] for k, v in preds.items()}
        if kind == "union":
            out.append(union(outputs, d.text, priority=tuple(spec.get("priority", (tc, llm)))))
        elif kind == "label_routed":
            out.append(
                label_routed(
                    outputs,
                    train_counts,
                    int(spec.get("tau", 10)),
                    tc,
                    llm,
                    both=set(spec.get("both", ())),
                    text=d.text,
                )
            )
        elif kind == "confidence_gated":
            cfg = GateConfig(
                theta_hi=float(spec.get("theta_hi", 0.9)),
                theta_lo=float(spec.get("theta_lo", 0.3)),
                tau=int(spec.get("tau", 10)),
                tc=tc,
                llm=llm,
            )
            out.append(confidence_gated(outputs, train_counts, cfg, text=d.text).spans)
        else:
            raise ValueError(f"unknown fusion {kind!r}")
    return out


# -- run -------------------------------------------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(config: Mapping[str, Any], out_dir: str | Path | None = None, limit: int | None = None) -> Path:
    cfg = dict(config)
    bench = cfg["benchmark"]
    docs = list(read_jsonl(Path(bench["path"]) / f"{bench.get('split', 'test')}.jsonl"))
    limit = limit if limit is not None else bench.get("limit")
    if limit:
        docs = docs[: int(limit)]
    train_counts: dict[str, int] = {}
    train_split = bench.get("train_split", "train")
    train_path = Path(bench["path"]) / f"{train_split}.jsonl"
    if train_path.exists():
        train_counts = dict(label_support(read_jsonl(train_path)))

    run_id = (
        time.strftime("%Y%m%d-%H%M%S")
        + "-"
        + hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()[:6]
    )
    out = Path(out_dir or cfg.get("output", "results")) / cfg.get("name", "run") / run_id
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # detectors
    llm_client = None
    preds: dict[str, list[list[Span]]] = {}
    base_names: list[str] = []
    for name, dcfg in (cfg.get("detectors") or {}).items():
        det = build_detector(name, dcfg, llm_client)
        if dcfg.get("type", name) == "llm":
            llm_client = det.client
        preds[name] = det.detect(docs)
        base_names.append(name)
        write_jsonl(
            out / "predictions" / f"{name}.jsonl",
            [Doc(d.doc_id, d.text, p, {}) for d, p in zip(docs, preds[name], strict=True)],
        )

    # fusion
    for spec in cfg.get("fusion") or []:
        fname = spec.get("as") or spec["name"]
        preds[fname] = apply_fusion(spec, docs, {k: preds[k] for k in base_names}, train_counts)
        write_jsonl(
            out / "predictions" / f"{fname}.jsonl",
            [Doc(d.doc_id, d.text, p, {}) for d, p in zip(docs, preds[fname], strict=True)],
        )

    # evaluation
    ev = cfg.get("eval") or {}
    modes = ev.get("modes", ["strict", "relaxed"])
    levels = ev.get("levels", ["subtype"])
    reference = ev.get("reference")
    metrics: dict[str, Any] = {"n_docs": len(docs), "train_counts": train_counts, "systems": {}}
    summary_lines = [
        f"# {cfg.get('name', 'run')} — {run_id}",
        "",
        f"{len(docs)} documents; train label counts from `{train_split}`.",
        "",
    ]
    for name, spans in preds.items():
        pred_map = {d.doc_id: p for d, p in zip(docs, spans, strict=True)}
        metrics["systems"][name] = {}
        for mode in modes:
            for level in levels:
                rep = evaluate(docs, pred_map, mode=mode, level=level)
                entry: dict[str, Any] = rep.to_dict()
                if level == "subtype" and train_counts:
                    entry["by_bucket"] = {b: v.to_dict() for b, v in by_bucket(rep.per_label, train_counts).items()}
                if ev.get("hipaa_view", True) and level == "subtype":
                    entry["hipaa"] = evaluate(docs, pred_map, mode=mode, level=level, hipaa_only=True).to_dict()
                if reference and reference in preds and reference != name and level == "subtype":
                    ref_map = {d.doc_id: p for d, p in zip(docs, preds[reference], strict=True)}
                    a = evaluate_docs(docs, pred_map, mode=mode, level=level)
                    b = evaluate_docs(docs, ref_map, mode=mode, level=level)
                    bs = ev.get("bootstrap") or {}
                    entry["vs_reference"] = {
                        m: paired_bootstrap(a, b, m, int(bs.get("n", 1000)), int(bs.get("seed", 0))).to_dict()
                        for m in ("recall", "precision", "f1")
                    }
                metrics["systems"][name][f"{mode}/{level}"] = entry
                if level == "subtype":
                    summary_lines += [f"## {name} ({mode})", "", report_markdown(rep), ""]
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # complementarity across base detectors
    if len(base_names) >= 2:
        comp = {}
        for mode in modes:
            c = complementarity(
                docs, {n: {d.doc_id: p for d, p in zip(docs, preds[n], strict=True)} for n in base_names}, mode=mode
            )
            comp[mode] = c.to_dict()
            o = c.overall
            summary_lines += [
                f"## complementarity ({mode})",
                "",
                f"recall: {o.recall}; oracle-union recall {o.oracle_union_recall:.3f}; intersection recall "
                f"{o.intersection_recall:.3f}; pairs: "
                + "; ".join(f"{p.a}/{p.b} Jaccard {p.jaccard:.3f}, McNemar p {p.mcnemar_p:.3g}" for p in o.pairs),
                "",
            ]
        (out / "complementarity.json").write_text(json.dumps(comp, indent=2) + "\n", encoding="utf-8")

    # cost
    cost = llm_client.cost_summary() if llm_client is not None else {}
    (out / "cost.json").write_text(json.dumps(cost, indent=2) + "\n", encoding="utf-8")
    if llm_client is not None:
        llm_client.write_log(out / "llm_calls.jsonl")
    (out / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ns = ap.parse_args(argv)
    out = run(load_config(ns.config), ns.out, ns.limit)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
