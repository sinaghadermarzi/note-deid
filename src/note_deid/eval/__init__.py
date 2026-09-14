"""Evaluation: matching semantics, entity/token/document metrics, frequency buckets, paired bootstrap,
error-complementarity statistics, and report writers. Semantics follow the i2b2-2014 evaluation script.
"""

from note_deid.eval.bootstrap import BootstrapResult, holm_correction, paired_bootstrap
from note_deid.eval.buckets import by_bucket, label_support
from note_deid.eval.complementarity import Complementarity, complementarity, miss_table
from note_deid.eval.matching import MATCH_MODES, MatchResult, match_spans
from note_deid.eval.metrics import LEVELS, PRF, DocCounts, Report, evaluate, evaluate_docs
from note_deid.eval.report import dump_json, report_markdown

__all__ = [
    "LEVELS",
    "MATCH_MODES",
    "PRF",
    "BootstrapResult",
    "Complementarity",
    "DocCounts",
    "MatchResult",
    "Report",
    "by_bucket",
    "complementarity",
    "dump_json",
    "evaluate",
    "evaluate_docs",
    "holm_correction",
    "label_support",
    "match_spans",
    "miss_table",
    "paired_bootstrap",
    "report_markdown",
]
