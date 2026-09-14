import pytest

from note_deid.eval import (
    by_bucket,
    complementarity,
    evaluate,
    evaluate_docs,
    holm_correction,
    label_support,
    match_spans,
    paired_bootstrap,
    report_markdown,
)
from note_deid.schema import Doc, Span

TEXT = "Pt John Smith (MRN 123456) seen by Dr. Lee on 03/04/2019 at Mercy Hospital. Age 92. Occupation: welder."


def gold_doc(doc_id="d1") -> Doc:
    spans = [
        Span(3, 13, "PATIENT"),  # John Smith
        Span(19, 25, "MEDICALRECORD"),  # 123456
        Span(39, 42, "DOCTOR"),  # Lee
        Span(46, 56, "DATE"),  # 03/04/2019
        Span(60, 74, "HOSPITAL"),  # Mercy Hospital
        Span(80, 82, "AGE"),  # 92
        Span(96, 102, "PROFESSION"),  # welder
    ]
    return Doc(doc_id, TEXT, spans).fill_span_text().validate()


# -- matching ------------------------------------------------------------------------------------------------------
def test_strict_relaxed_and_typeless_matching():
    gold = [Span(3, 13, "PATIENT"), Span(46, 56, "DATE")]
    pred = [Span(8, 13, "PATIENT"), Span(46, 56, "DOCTOR")]  # partial name, mislabeled date
    assert match_spans(gold, pred, "strict").counts == (0, 2, 2)
    assert match_spans(gold, pred, "relaxed").counts == (1, 1, 1)
    assert match_spans(gold, pred, "relaxed_typeless").counts == (2, 0, 0)


def test_matching_is_one_to_one_and_prefers_exact_extent():
    gold = [Span(0, 4, "NAME"), Span(5, 9, "NAME")]
    pred = [Span(0, 9, "NAME")]  # one prediction spanning two gold names
    r = match_spans(gold, pred, "relaxed")
    assert r.counts == (1, 0, 1)
    gold = [Span(0, 4, "NAME")]
    pred = [Span(0, 9, "NAME"), Span(0, 4, "NAME")]
    r = match_spans(gold, pred, "relaxed")
    assert r.tp[0][1] == Span(0, 4, "NAME")  # exact extent chosen over the larger overlap
    assert r.counts == (1, 1, 0)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        match_spans([], [], "loose")


# -- evaluate ------------------------------------------------------------------------------------------------------
def test_evaluate_levels_and_counts():
    g = gold_doc()
    pred = [
        Span(3, 13, "PATIENT"),
        Span(19, 25, "MEDICALRECORD"),
        Span(46, 56, "DATE"),
        Span(60, 74, "ORGANIZATION"),  # wrong subtype, same category
        Span(88, 95, "PROFESSION"),  # spurious
    ]
    rep = evaluate([g], {"d1": pred}, mode="strict", level="subtype")
    assert rep.micro.tp == 3 and rep.micro.fp == 2 and rep.micro.fn == 4
    assert rep.per_label["HOSPITAL"].fn == 1 and rep.per_label["ORGANIZATION"].fp == 1
    assert rep.doc_leak_rate == 1.0 and rep.docs_with_gold == 1
    cat = evaluate([g], {"d1": pred}, mode="strict", level="category")
    assert cat.per_label["LOCATION"].tp == 1  # ORGANIZATION and HOSPITAL both map to LOCATION
    binary = evaluate([g], {"d1": pred}, mode="strict", level="binary")
    assert set(binary.per_label) == {"PHI"} and binary.micro.tp == 4
    assert 0 < rep.token.recall < 1 and rep.token.tp > 0


def test_hipaa_view_applies_age_rule_and_drops_non_hipaa_labels():
    g = gold_doc()
    rep = evaluate([g], {"d1": list(g.spans)}, hipaa_only=True)
    # DOCTOR, HOSPITAL, PROFESSION are not HIPAA identifiers; AGE 92 is (>= 90)
    assert set(rep.per_label) == {"PATIENT", "MEDICALRECORD", "DATE", "AGE"}
    young = Doc("d2", "Age 45 male.", [Span(4, 6, "AGE")]).fill_span_text()
    assert evaluate([young], {"d2": list(young.spans)}, hipaa_only=True).micro.support == 0


def test_missing_predictions_count_as_misses_and_empty_docs_do_not_leak():
    g = gold_doc()
    empty = Doc("d0", "nothing here", [])
    rep = evaluate([g, empty], {})
    assert rep.micro.fn == 7 and rep.n_docs == 2
    assert rep.doc_leak_rate == 0.5 and rep.leak_rate_among_docs_with_gold == 1.0
    assert "PATIENT" in report_markdown(rep)


def test_macro_averages_only_supported_labels():
    g = gold_doc()
    rep = evaluate([g], {"d1": [Span(3, 13, "PATIENT"), Span(0, 2, "URL")]})  # URL has no gold support
    assert rep.macro_recall == pytest.approx(1 / 7)
    assert rep.per_label["URL"].fp == 1


# -- buckets -------------------------------------------------------------------------------------------------------
def test_buckets_group_labels_by_training_frequency():
    g = gold_doc()
    rep = evaluate([g], {"d1": list(g.spans)})
    train_counts = {"PATIENT": 500, "DATE": 900, "DOCTOR": 120, "MEDICALRECORD": 30, "HOSPITAL": 5}
    b = by_bucket(rep.per_label, train_counts)
    assert b[">200"].tp == 2 and b["51-200"].tp == 1 and b["11-50"].tp == 1 and b["1-10"].tp == 1
    assert b["0"].tp == 2  # AGE and PROFESSION absent from training
    assert label_support([g])["DATE"] == 1


# -- bootstrap -----------------------------------------------------------------------------------------------------
def test_paired_bootstrap_is_deterministic_and_detects_improvement():
    docs = [gold_doc(f"d{i}") for i in range(6)]
    full = {d.doc_id: list(d.spans) for d in docs}
    half = {d.doc_id: list(d.spans)[:3] for d in docs}
    a = evaluate_docs(docs, full)
    b = evaluate_docs(docs, half)
    r1 = paired_bootstrap(a, b, metric="recall", n_resamples=200, seed=1)
    r2 = paired_bootstrap(a, b, metric="recall", n_resamples=200, seed=1)
    assert r1.to_dict() == r2.to_dict()
    assert r1.delta == pytest.approx(4 / 7) and r1.ci_low > 0 and r1.p_value < 0.05
    same = paired_bootstrap(a, a, metric="f1", n_resamples=50, seed=0)
    assert same.delta == 0 and same.ci_low == 0 == same.ci_high and same.p_value == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[:-1])


def test_holm_correction():
    # sorted p: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone: 0.03, 0.06, 0.06 (returned in input order)
    assert holm_correction([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


# -- complementarity -----------------------------------------------------------------------------------------------
def test_complementarity_on_toy_systems():
    g = gold_doc()
    s = list(g.spans)  # 7 gold spans
    # A misses spans 0 and 1; B misses spans 1 and 2; both catch the rest. B adds one false positive.
    pred_a = {"d1": s[2:]}
    pred_b = {"d1": [s[0]] + s[3:] + [Span(88, 95, "PROFESSION")]}
    c = complementarity([g], {"A": pred_a, "B": pred_b}, mode="strict")
    o = c.overall
    assert o.n_gold == 7
    assert o.recall == {"A": pytest.approx(5 / 7), "B": pytest.approx(5 / 7)}
    assert o.oracle_union_recall == pytest.approx(6 / 7)  # only span 1 missed by both
    assert o.intersection_recall == pytest.approx(4 / 7)
    pair = o.pairs[0]
    assert (pair.fn_a, pair.fn_b, pair.fn_both, pair.fn_either) == (2, 2, 1, 3)
    assert pair.jaccard == pytest.approx(1 / 3)
    assert pair.p_b_missed_given_a_missed == pytest.approx(0.5) and pair.p_b_missed == pytest.approx(2 / 7)
    assert (pair.mcnemar_b, pair.mcnemar_c) == (1, 1) and pair.mcnemar_p == 1.0
    assert set(c.by_group) == {sp.label for sp in s}
    assert c.by_group["PATIENT"].recall == {"A": 0.0, "B": 1.0}
    # agreement precision: A's predictions confirmed by B are s[3:] (4 spans), all true
    assert c.agreement_precision["A"] == (4, 4)
    # B's predictions confirmed by A: s[3:] (the FP and s[0] are not predicted by A)
    assert c.agreement_precision["B"] == (4, 4)
    d = c.to_dict()
    assert d["overall"]["pairs"][0]["kappa"] == pytest.approx(pair.kappa)
