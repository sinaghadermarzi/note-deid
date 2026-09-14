"""Error-complementarity statistics between detection systems (RQ1, framework-designs.md §2 and §4).

Every gold span gets a miss indicator per system. From those we derive: per-system recall, false-negative set
overlap (Jaccard), conditional vs marginal miss probabilities, oracle-union recall (a span counts as found if any
system found it), intersection recall (found by all), McNemar counts, Cohen's kappa on the miss indicators, and
agreement precision (predictions confirmed by every other system). All statistics are also available per group
(label, frequency bucket, note type ...).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from note_deid.eval.matching import match_spans, span_key, spans_match
from note_deid.eval.metrics import prepare_spans
from note_deid.schema import Doc, Span


@dataclass
class MissRow:
    doc_id: str
    span: Span
    missed: dict[str, bool]  # system -> missed?
    group: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def miss_table(
    gold_docs: Iterable[Doc],
    predictions: Mapping[str, Mapping[str, Sequence[Span]]],
    mode: str = "strict",
    level: str = "subtype",
    hipaa_only: bool = False,
    group_by: Callable[[Doc, Span], str] | None = None,
) -> list[MissRow]:
    """One row per gold span with a miss flag for every system in ``predictions`` ({system: {doc_id: spans}})."""
    systems = list(predictions)
    rows: list[MissRow] = []
    for doc in gold_docs:
        gold = prepare_spans(doc.spans, doc.text, level, hipaa_only)
        fn_sets: dict[str, set[tuple[int, int, str]]] = {}
        for sys_name in systems:
            pred = prepare_spans(predictions[sys_name].get(doc.doc_id, []), doc.text, level, hipaa_only)
            fn_sets[sys_name] = {span_key(s) for s in match_spans(gold, pred, mode).fn}
        for g in gold:
            key = span_key(g)
            rows.append(
                MissRow(
                    doc.doc_id,
                    g,
                    {s: key in fn_sets[s] for s in systems},
                    group_by(doc, g) if group_by else g.label,
                    dict(doc.meta),
                )
            )
    return rows


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 degree of freedom."""
    return math.erfc(math.sqrt(x / 2.0)) if x > 0 else 1.0


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Continuity-corrected McNemar statistic and p-value for discordant counts b (A misses, B catches) and c."""
    if b + c == 0:
        return 0.0, 1.0
    stat = max(0, abs(b - c) - 1) ** 2 / (b + c)
    return stat, _chi2_sf_1df(stat)


def cohen_kappa(x: Sequence[bool], y: Sequence[bool]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(x, y, strict=True) if a == b) / n
    px1 = sum(x) / n
    py1 = sum(y) / n
    pe = px1 * py1 + (1 - px1) * (1 - py1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


@dataclass
class PairStats:
    a: str
    b: str
    fn_a: int
    fn_b: int
    fn_both: int
    fn_either: int
    jaccard: float
    p_b_missed_given_a_missed: float
    p_b_missed: float
    p_a_missed_given_b_missed: float
    p_a_missed: float
    mcnemar_b: int  # a misses, b catches
    mcnemar_c: int  # a catches, b misses
    mcnemar_stat: float
    mcnemar_p: float
    kappa: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class GroupStats:
    n_gold: int
    recall: dict[str, float]
    oracle_union_recall: float
    intersection_recall: float
    pairs: list[PairStats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_gold": self.n_gold,
            "recall": {k: round(v, 6) for k, v in self.recall.items()},
            "oracle_union_recall": round(self.oracle_union_recall, 6),
            "intersection_recall": round(self.intersection_recall, 6),
            "pairs": [p.to_dict() for p in self.pairs],
        }


def _stats(rows: Sequence[MissRow], systems: Sequence[str]) -> GroupStats:
    n = len(rows)
    if n == 0:
        return GroupStats(0, dict.fromkeys(systems, 0.0), 0.0, 0.0, [])
    recall = {s: 1 - sum(r.missed[s] for r in rows) / n for s in systems}
    missed_by_all = sum(1 for r in rows if all(r.missed[s] for s in systems))
    missed_by_any = sum(1 for r in rows if any(r.missed[s] for s in systems))
    pairs: list[PairStats] = []
    for i, a in enumerate(systems):
        for b in systems[i + 1 :]:
            ma = [r.missed[a] for r in rows]
            mb = [r.missed[b] for r in rows]
            fn_a, fn_b = sum(ma), sum(mb)
            both = sum(1 for x, y in zip(ma, mb, strict=True) if x and y)
            either = sum(1 for x, y in zip(ma, mb, strict=True) if x or y)
            bcount = sum(1 for x, y in zip(ma, mb, strict=True) if x and not y)
            ccount = sum(1 for x, y in zip(ma, mb, strict=True) if y and not x)
            stat, p = mcnemar(bcount, ccount)
            pairs.append(
                PairStats(
                    a=a,
                    b=b,
                    fn_a=fn_a,
                    fn_b=fn_b,
                    fn_both=both,
                    fn_either=either,
                    jaccard=both / either if either else 0.0,
                    p_b_missed_given_a_missed=both / fn_a if fn_a else 0.0,
                    p_b_missed=fn_b / n,
                    p_a_missed_given_b_missed=both / fn_b if fn_b else 0.0,
                    p_a_missed=fn_a / n,
                    mcnemar_b=bcount,
                    mcnemar_c=ccount,
                    mcnemar_stat=stat,
                    mcnemar_p=p,
                    kappa=cohen_kappa(ma, mb),
                )
            )
    return GroupStats(n, recall, 1 - missed_by_all / n, 1 - missed_by_any / n, pairs)


@dataclass
class Complementarity:
    systems: list[str]
    mode: str
    level: str
    overall: GroupStats
    by_group: dict[str, GroupStats]
    agreement_precision: dict[str, tuple[int, int]]  # system -> (tp among agreed predictions, agreed predictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "systems": self.systems,
            "mode": self.mode,
            "level": self.level,
            "overall": self.overall.to_dict(),
            "by_group": {g: s.to_dict() for g, s in self.by_group.items()},
            "agreement_precision": {
                s: {"tp": tp, "n": n, "precision": round(tp / n, 6) if n else 0.0}
                for s, (tp, n) in self.agreement_precision.items()
            },
        }


def agreement_precision(
    gold_docs: Iterable[Doc],
    predictions: Mapping[str, Mapping[str, Sequence[Span]]],
    mode: str = "strict",
    level: str = "subtype",
    hipaa_only: bool = False,
) -> dict[str, tuple[int, int]]:
    """For each system: among its predictions that every other system also predicts (under ``mode``), how many are
    true positives. High agreement precision means intersection-style fusion is safe."""
    systems = list(predictions)
    out: dict[str, list[int]] = {s: [0, 0] for s in systems}
    for doc in gold_docs:
        gold = prepare_spans(doc.spans, doc.text, level, hipaa_only)
        preds = {s: prepare_spans(predictions[s].get(doc.doc_id, []), doc.text, level, hipaa_only) for s in systems}
        for s in systems:
            tp_keys = {span_key(p) for _, p in match_spans(gold, preds[s], mode).tp}
            for p in preds[s]:
                agreed = all(any(spans_match(p, q, mode) for q in preds[o]) for o in systems if o != s)
                if agreed:
                    out[s][1] += 1
                    out[s][0] += int(span_key(p) in tp_keys)
    return {s: (v[0], v[1]) for s, v in out.items()}


def complementarity(
    gold_docs: Iterable[Doc],
    predictions: Mapping[str, Mapping[str, Sequence[Span]]],
    mode: str = "strict",
    level: str = "subtype",
    hipaa_only: bool = False,
    group_by: Callable[[Doc, Span], str] | None = None,
) -> Complementarity:
    gold_list = list(gold_docs)
    systems = list(predictions)
    rows = miss_table(gold_list, predictions, mode, level, hipaa_only, group_by)
    groups: dict[str, list[MissRow]] = defaultdict(list)
    for r in rows:
        groups[r.group].append(r)
    return Complementarity(
        systems=systems,
        mode=mode,
        level=level,
        overall=_stats(rows, systems),
        by_group={g: _stats(rs, systems) for g, rs in sorted(groups.items())},
        agreement_precision=agreement_precision(gold_list, predictions, mode, level, hipaa_only),
    )
