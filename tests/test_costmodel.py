from note_deid.llm.costmodel import DESIGN_BY_ID, PRICE_POINTS, Workload, estimate, main, markdown_table

PRICES = {"strong": PRICE_POINTS["sonnet-5"], "cheap": PRICE_POINTS["haiku-5.5"]}


def test_per_candidate_designs_cost_more_than_batched_ones():
    w = Workload()
    d1, d1n, d2, d4, d6, f1 = (estimate(i, w) for i in ("D1", "D1n", "D2", "D4", "D6", "F1"))
    assert d1.calls == d1n.calls == w.candidates and d2.calls == 1
    assert d1.usd(PRICES) > 4 * d2.usd(PRICES)  # ~6x with cache reads; far more without
    assert d1n.cached_tokens == w.candidates * (w.guideline_tokens + w.note_tokens)
    assert d4.usd(PRICES) < d1.usd(PRICES)  # one output token instead of a verdict
    assert d6.usd(PRICES) < d2.usd(PRICES) < 3 * f1.usd(PRICES)
    assert d6.output_tokens == d2.output_tokens / w.repetition and d6.input_tokens > d2.input_tokens


def test_cache_and_batch_only_lower_cost_and_cache_helps_per_candidate_most():
    w = Workload()
    for i in DESIGN_BY_ID:
        e = estimate(i, w)
        assert e.usd(PRICES, cache=False) >= e.usd(PRICES, cache=True) >= e.usd(PRICES, cache=True, batch=True)
    d1, d2 = estimate("D1", w), estimate("D2", w)
    assert d1.usd(PRICES, cache=False) / d1.usd(PRICES) > d2.usd(PRICES, cache=False) / d2.usd(PRICES)


def test_gated_and_mixed_designs_split_cost_by_role():
    w = Workload(flagged_fraction=0.5, flagged_candidate_fraction=0.5)
    d3, d3c, d7 = estimate("D3", w), estimate("D3c", w), estimate("D7", w)
    assert {p.role for p in d3.parts} == {"cheap", "strong"} and [p.role for p in d3c.parts] == ["strong"]
    assert d3c.usd(PRICES) < d3.usd(PRICES)
    assert d3c.output_tokens == 0.5 * estimate("D2", w).output_tokens
    cheap_only = {"strong": PRICE_POINTS["local"], "cheap": PRICE_POINTS["haiku-5.5"]}
    assert d7.usd(cheap_only) > 0 and estimate("D2", w).usd(cheap_only) == 0
    # the cheap role priced like the strong one (the current llm-fast alias) makes the cascades dearer than F1
    same = {"strong": PRICE_POINTS["sonnet-5"], "cheap": PRICE_POINTS["sonnet-5"]}
    assert d3.usd(same) > estimate("F1", w).usd(same) and d7.usd(same) > estimate("F1", w).usd(same)


def test_derived_proposals_and_proposer_free_control():
    w = Workload(phi_spans=40)
    assert w.k_fast == 60 and Workload(fast_proposals=25).k_fast == 25
    d8 = estimate("D8", w)
    assert d8.calls == 1 and d8.input_tokens == w.note_tokens * (1 + w.token_index_overhead)
    assert d8.output_tokens == w.phi_spans * w.phi_tokens_per_span * w.token_label_tokens


def test_workload_scaling_and_table_rendering():
    short, long = Workload(note_tokens=500, candidates=40), Workload(note_tokens=3000, candidates=240)
    for i in ("F1", "D1", "D2", "D6", "D8"):
        assert estimate(i, long).usd(PRICES) > estimate(i, short).usd(PRICES)
    assert estimate("D5", short).parts == [] and estimate("D5", short).usd(PRICES) == 0
    table = markdown_table(Workload(), PRICES, notes=100)
    lines = table.splitlines()
    assert lines[0].startswith("| ID | Design | local compute |") and len(lines) == 2 + len(DESIGN_BY_ID)
    assert "| F1 |" in table and "xF1" in lines[0]
    out = main(["--note-tokens", "2000", "--strong", "opus-5", "--notes", "10"])
    assert "N=2000" in out and "claude-opus-5" in out and "| D2 |" in out
    out2 = main(["--input-per-m", "1.25", "--output-per-m", "10", "--cache-read-mult", "0.1", "--cheap", "sonnet-5"])
    assert "strong=custom (1.25/10.0" in out2 and "cheap=claude-sonnet-5" in out2
