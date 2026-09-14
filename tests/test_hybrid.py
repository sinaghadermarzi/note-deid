from note_deid.hybrid import (
    GateConfig,
    confidence_gated,
    label_routed,
    merge_overlapping,
    resolve_conflicts,
    risk_features,
    risk_score,
    select_for_llm,
    union,
)
from note_deid.schema import Span

TEXT = "Pt John Smith (MRN 123456) seen by Dr. Lee on 03/04/2019. Email j.smith@example.org. Fax 555-0100."


def tc(start, end, label, score):
    return Span(start, end, label, TEXT[start:end], "tc", score)


def llm(start, end, label):
    return Span(start, end, label, TEXT[start:end], "llm")


def test_merge_overlapping_same_label_union_extent():
    merged = merge_overlapping([tc(3, 7, "PATIENT", 0.9), llm(3, 13, "PATIENT"), llm(46, 56, "DATE")], TEXT)
    assert merged[0] == Span(3, 13, "PATIENT", "John Smith", "llm+tc", 0.9)
    assert len(merged) == 2


def test_resolve_conflicts_prefers_priority_unless_below_min_score():
    a, b = tc(60, 79, "URL", 0.4), llm(60, 79, "EMAIL")
    assert resolve_conflicts([a, b], priority=("tc", "llm")) == [a]
    assert resolve_conflicts([a, b], priority=("tc", "llm"), min_score={"tc": 0.5}) == [b]


def test_union_is_superset_of_both_systems():
    out = {"tc": [tc(3, 13, "PATIENT", 0.95), tc(19, 25, "MEDICALRECORD", 0.8)], "llm": [llm(39, 42, "DOCTOR")]}
    fused = union(out, TEXT)
    assert [(s.start, s.end, s.label) for s in fused] == [
        (3, 13, "PATIENT"),
        (19, 25, "MEDICALRECORD"),
        (39, 42, "DOCTOR"),
    ]


def test_label_routed_uses_frequency_and_support():
    out = {
        "tc": [tc(3, 13, "PATIENT", 0.95), tc(39, 42, "DOCTOR", 0.7), tc(66, 84, "EMAIL", 0.6)],
        "llm": [llm(3, 13, "PATIENT"), llm(46, 56, "DATE"), llm(66, 84, "EMAIL"), llm(90, 98, "FAX")],
    }
    counts = {"PATIENT": 500, "DOCTOR": 300, "DATE": 900, "EMAIL": 3, "FAX": 0}
    fused = label_routed(out, counts, tau=10)
    labels = {(s.start, s.label): s.source for s in fused}
    assert labels[(3, "PATIENT")] == "tc"  # frequent label -> TC only
    assert labels[(66, "EMAIL")] == "llm"  # rare label -> LLM only
    assert labels[(90, "FAX")] == "llm"
    assert (46, "DATE") not in labels  # frequent label predicted only by the LLM is not taken
    both = label_routed(out, counts, tau=10, both={"DATE"})
    assert any(s.label == "DATE" for s in both)


def test_confidence_gated_rules_and_verifier_calls():
    out = {
        "tc": [tc(3, 13, "PATIENT", 0.97), tc(19, 25, "MEDICALRECORD", 0.5), tc(39, 42, "DOCTOR", 0.2)],
        "llm": [llm(3, 13, "PATIENT"), llm(46, 56, "DATE"), llm(66, 84, "EMAIL"), llm(90, 98, "FAX")],
    }
    counts = {"PATIENT": 500, "DOCTOR": 300, "DATE": 900, "MEDICALRECORD": 200, "EMAIL": 3}
    verified = {(19, 25): True, (39, 42): False, (46, 56): True}
    res = confidence_gated(
        out, counts, GateConfig(theta_hi=0.9, theta_lo=0.3, tau=10), verifier=lambda s: verified[(s.start, s.end)]
    )
    rules = {(d.span.start, d.span.end): (d.rule, d.accepted) for d in res.decisions}
    assert rules[(3, 13)] == ("agree", True)
    assert rules[(19, 25)] == ("tc-only:low:verified", True)
    assert rules[(39, 42)] == ("tc-only:low:verified", False)
    assert rules[(46, 56)] == ("llm-only:frequent:verified", True)
    assert rules[(66, 84)] == ("llm-only:rare-or-unsupported", True)
    assert rules[(90, 98)] == ("llm-only:rare-or-unsupported", True)  # FAX absent from training counts
    assert res.verifier_calls == 3
    assert [(s.start, s.label) for s in res.spans] == [
        (3, "PATIENT"),
        (19, "MEDICALRECORD"),
        (46, "DATE"),
        (66, "EMAIL"),
        (90, "FAX"),
    ]
    # no verifier: low-confidence TC-only and frequent LLM-only spans are dropped
    res2 = confidence_gated(out, counts, GateConfig(theta_hi=0.9, theta_lo=0.3, tau=10))
    assert [(s.start, s.label) for s in res2.spans] == [(3, "PATIENT"), (66, "EMAIL"), (90, "FAX")]
    # forced TC probability rescues a frequent LLM-only span
    res3 = confidence_gated(out, counts, GateConfig(), tc_forced_prob=lambda s: 0.6 if s.label == "DATE" else 0.0)
    assert any(s.label == "DATE" for s in res3.spans)


def test_risk_features_and_selection():
    tc_spans = [tc(3, 13, "PATIENT", 0.97)]
    f = risk_features(TEXT, tc_spans, token_max_probs=[0.99, 0.5, 0.6, 0.99], lexicon={"pt", "email", "fax"})
    assert f.uncertainty_mass == 0.5
    assert f.uncovered_cues >= 2  # the email and the fax number are not covered by TC spans
    assert f.capitalized_oov >= 1  # "Lee" (Dr. -> sentence start heuristic may or may not fire on "Dr")
    covered = risk_features(TEXT, tc_spans + [tc(66, 84, "EMAIL", 0.9), tc(90, 98, "FAX", 0.9)], lexicon={"pt"})
    assert covered.uncovered_cues < f.uncovered_cues
    assert risk_score(f) > risk_score(covered)
    chosen = select_for_llm({"a": f, "b": covered}, rho=risk_score(covered))
    assert chosen == {"a"}
