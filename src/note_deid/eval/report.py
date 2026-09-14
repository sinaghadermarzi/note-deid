"""Serialization helpers for evaluation outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from note_deid.eval.metrics import PRF, Report


def dump_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.to_dict() if hasattr(obj, "to_dict") else obj
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def prf_table(per_label: Mapping[str, PRF], title: str = "label") -> str:
    lines = [f"| {title} | support | P | R | F1 |", "|---|---:|---:|---:|---:|"]
    for label, v in per_label.items():
        lines.append(f"| {label} | {v.support} | {v.precision:.3f} | {v.recall:.3f} | {v.f1:.3f} |")
    return "\n".join(lines)


def report_markdown(report: Report) -> str:
    head = (
        f"**{report.mode} / {report.level}{' / HIPAA-only' if report.hipaa_only else ''}** — "
        f"{report.n_docs} docs; micro P {report.micro.precision:.3f} R {report.micro.recall:.3f} "
        f"F1 {report.micro.f1:.3f}; macro F1 {report.macro_f1:.3f}; doc leak rate {report.doc_leak_rate:.3f}"
    )
    if report.token:
        head += f"; token P {report.token.precision:.3f} R {report.token.recall:.3f}"
    return head + "\n\n" + prf_table(report.per_label)
