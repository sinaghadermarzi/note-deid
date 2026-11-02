from note_deid.llm.costmodel import DESIGN_BY_ID, PRICE_POINTS, Workload, estimate, main, markdown_table

PRICES = {"strong": PRICE_POINTS["sonnet-5"], "fast": PRICE_POINTS["haiku-5.5"]}


def test_per_candidate_designs_cost_more_than_batched_ones():
    w = Workload()
    d1, d2, d4, d6, f1 = (estimate(i, w) for i in ("D1", "D2", "D4", "D6", "F1"))
    assert d1.calls == w.candidates and d2.calls == 1
    assert d1.usd(PRICES) > 4 * d2.usd(PRICES)  # ~4.7x with cache reads; ~19x without
    assert d4.usd(PRICES) < d1.usd(PRICES)  # one output token instead of a JSON decision
    assert d6.usd(PRICES) < d2.usd(PRICES) < 3 * f1.usd(PRICES)  # entity grouping only shrinks the output
    assert d6.input_tokens == d2.input_tokens


def test_cache_and_batch_only_lower_cost_and_cache_helps_per_candidate_most():
    w = Workload()
    for i in DESIGN_BY_ID:
        e = estimate(i, w)
        assert e.usd(PRICES, cache=False) >= e.usd(PRICES, cache=True) >= e.usd(PRICES, cache=True, batch=True)
    d1, d2 = estimate("D1", w), estimate("D2", w)
    assert d1.usd(PRICES, cache=False) / d1.usd(PRICES) > d2.usd(PRICES, cache=False) / d2.usd(PRICES)


def test_gated_and_mixed_designs_split_cost_by_role():
    w = Workload(flagged_fraction=0.5)
    d3, d3c, d7 = estimate("D3", w), estimate("D3c", w), estimate("D7", w)
    assert {p.role for p in d3.parts} == {"fast", "strong"} and [p.role for p in d3c.parts] == ["strong"]
    assert d3c.usd(PRICES) < d3.usd(PRICES)
    assert d3c.output_tokens == 0.5 * estimate("D2", w).output_tokens
    fast_only = {"strong": PRICE_POINTS["local"], "fast": PRICE_POINTS["haiku-5.5"]}
    assert d7.usd(fast_only) > 0 and estimate("D2", w).usd(fast_only) == 0


def test_workload_scaling_and_table_rendering():
    short, long = Workload(note_tokens=500, candidates=40), Workload(note_tokens=3000, candidates=240)
    for i in ("F1", "D1", "D2", "D6"):
        assert estimate(i, long).usd(PRICES) > estimate(i, short).usd(PRICES)
    assert estimate("D5", short).parts == [] and estimate("D5", short).usd(PRICES) == 0
    table = markdown_table(Workload(), PRICES, notes=100)
    lines = table.splitlines()
    assert lines[0].startswith("| ID |") and len(lines) == 2 + len(DESIGN_BY_ID)
    assert "| F1 |" in table and "x" in lines[0]
    out = main(["--note-tokens", "2000", "--strong", "opus-5", "--notes", "10"])
    assert "N=2000" in out and "claude-opus-5" in out and "| D2 |" in out
    out2 = main(["--input-per-m", "1.25", "--output-per-m", "10", "--cache-read-mult", "0.1"])
    assert "strong=custom (1.25/10.0" in out2
