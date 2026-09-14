"""Fusion policies and routing: H1 union, H2 label-routed, H3 confidence-gated arbitration, H7 selective invocation.
Interface: each policy maps ``{system: [Span]}`` for one document to the fused span list.
"""

from note_deid.hybrid.fusion import (
    Decision,
    GateConfig,
    GateResult,
    confidence_gated,
    label_routed,
    merge_overlapping,
    resolve_conflicts,
    union,
)
from note_deid.hybrid.routing import RiskFeatures, risk_features, risk_score, select_for_llm

__all__ = [
    "Decision",
    "GateConfig",
    "GateResult",
    "RiskFeatures",
    "confidence_gated",
    "label_routed",
    "merge_overlapping",
    "resolve_conflicts",
    "risk_features",
    "risk_score",
    "select_for_llm",
    "union",
]
