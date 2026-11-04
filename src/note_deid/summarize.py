"""Flatten a run's ``metrics.json`` (and ``complementarity.json``) into the summary tables kept in ``results/summary/``.

    python -m note_deid.summarize results/<exp>/<run-id> --out results/summary/<exp>

Writes ``<out>.csv`` (one row per system x mode x level x view, with the paired-bootstrap delta against the run's
reference system when present) and ``<out>.complementarity.csv`` (one row per detector pair and mode). Every number is
copied from the run directory; nothing is recomputed, so a summary can always be traced back to its run id.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SYSTEM_COLUMNS: tuple[str, ...] = (
    "run_id",
    "system",
    "mode",
    "level",
    "view",
    "n_docs",
    "support",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "doc_leak_rate",
    "reference",
    "delta_recall",
    "delta_recall_ci_low",
    "delta_recall_ci_high",
    "delta_recall_p",
)
PAIR_COLUMNS: tuple[str, ...] = (
    "run_id",
    "mode",
    "n_gold",
    "oracle_union_recall",
    "intersection_recall",
    "a",
    "b",
    "recall_a",
    "recall_b",
    "fn_a",
    "fn_b",
    "fn_both",
    "fn_either",
    "jaccard",
    "p_b_missed_given_a_missed",
    "p_a_missed_given_b_missed",
    "mcnemar_p",
    "kappa",
)


def _entry_row(run_id: str, system: str, key: str, entry: Mapping[str, Any], view: str) -> dict[str, Any]:
    mode, _, level = key.partition("/")
    micro = entry["micro"]
    row: dict[str, Any] = {
        "run_id": run_id,
        "system": system,
        "mode": mode,
        "level": level,
        "view": view,
        "n_docs": entry["n_docs"],
        "support": micro["support"],
        "tp": micro["tp"],
        "fp": micro["fp"],
        "fn": micro["fn"],
        "precision": micro["precision"],
        "recall": micro["recall"],
        "f1": micro["f1"],
        "macro_f1": entry.get("macro_f1", entry.get("macro", {}).get("f1")),
        "doc_leak_rate": entry["doc_leak_rate"],
    }
    return row


def system_rows(metrics: Mapping[str, Any], run_id: str, reference: str | None = None) -> list[dict[str, Any]]:
    """One row per system x ``mode/level`` entry, plus a ``hipaa`` view row where the run computed one."""
    rows: list[dict[str, Any]] = []
    for system, entries in metrics["systems"].items():
        for key, entry in entries.items():
            row = _entry_row(run_id, system, key, entry, "full")
            vs = (entry.get("vs_reference") or {}).get("recall")
            if vs:
                row.update(
                    reference=reference,
                    delta_recall=vs["delta"],
                    delta_recall_ci_low=vs["ci_low"],
                    delta_recall_ci_high=vs["ci_high"],
                    delta_recall_p=vs["p_value"],
                )
            rows.append(row)
            if entry.get("hipaa"):
                rows.append(_entry_row(run_id, system, key, entry["hipaa"], "hipaa"))
    return rows


def pair_rows(comp: Mapping[str, Any], run_id: str) -> list[dict[str, Any]]:
    """One row per detector pair and matching mode from ``complementarity.json``."""
    rows: list[dict[str, Any]] = []
    for mode, c in comp.items():
        o = c["overall"]
        for p in o["pairs"]:
            rows.append(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "n_gold": o["n_gold"],
                    "oracle_union_recall": o["oracle_union_recall"],
                    "intersection_recall": o["intersection_recall"],
                    "a": p["a"],
                    "b": p["b"],
                    "recall_a": o["recall"][p["a"]],
                    "recall_b": o["recall"][p["b"]],
                    **{k: p[k] for k in PAIR_COLUMNS if k in p},
                }
            )
    return rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    return path


def summarize_run(run_dir: str | Path, out: str | Path) -> list[Path]:
    """Summarize one run directory; returns the files written (``<out>.csv`` and, if the run has complementarity
    statistics, ``<out>.complementarity.csv``)."""
    run_dir = Path(run_dir)
    out = Path(out)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config_path = run_dir / "config.yaml"
    reference = None
    if config_path.exists():
        import yaml

        reference = ((yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("eval") or {}).get("reference")
    written = [write_csv(out.with_suffix(".csv"), system_rows(metrics, run_dir.name, reference), SYSTEM_COLUMNS)]
    comp_path = run_dir / "complementarity.json"
    if comp_path.exists():
        comp = json.loads(comp_path.read_text(encoding="utf-8"))
        written.append(write_csv(out.with_suffix(".complementarity.csv"), pair_rows(comp, run_dir.name), PAIR_COLUMNS))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="results/<exp>/<run-id>")
    ap.add_argument("--out", required=True, help="output path without extension, e.g. results/summary/e1")
    ns = ap.parse_args(argv)
    for p in summarize_run(ns.run_dir, ns.out):
        print(p)


if __name__ == "__main__":
    main()
