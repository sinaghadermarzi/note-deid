"""Closed-form per-note token and cost model for the LLM detection designs: extraction formats F1/F3, the H5 editor,
and the decision-model family D1-D8 of framework-designs.md section 3.6 (table reproduced in Appendix D).

No API calls are made. A ``Workload`` describes a typical note (tokens, candidates, segments, ...), a ``Design`` turns
it into per-role ``Part`` records (requests, uncached input, cache-read input, output tokens), and ``Prices`` turn parts
into USD. Everything is a parameter so the comparison can be re-run once real counts are logged by ``LLMClient``:

    python -m note_deid.llm.costmodel                      # default workload, strong = sonnet-5, cheap = haiku-5.5
    python -m note_deid.llm.costmodel --note-tokens 3000 --candidates 240 --phi-spans 60 --segments 75
    python -m note_deid.llm.costmodel --cheap sonnet-5     # the cheap role priced like the current `llm-fast` alias
    python -m note_deid.llm.costmodel --input-per-m 1.25 --output-per-m 10 --cache-read-mult 0.1  # custom prices
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Workload:
    """Per-note quantities. Defaults are planning estimates for a ~1k-token note; replace them with logged counts
    (token counts of rendered prompts and outputs, candidate and repetition counts from the D0 analysis)."""

    note_tokens: int = 1000  # N: the note
    guideline_tokens: int = 1500  # G: system prompt = guideline + output-format instructions (cacheable prefix)
    gate_tokens: int = 600  # D3 gate system prompt, kept above the 512-token cacheable minimum
    phi_spans: int = 20  # P: gold PHI mentions per note
    candidates: int = 80  # K: candidates from the surface-form proposer (regex shapes, capitalized runs, numbers)
    fast_proposals: int = 0  # K': candidates from the over-generating cheap pass (D7); 0 = overgeneration * P
    overgeneration: float = 1.5  # the D7 proposer lists this multiple of P
    repetition: float = 1.6  # occurrences per distinct candidate string (D6 decides once per string; D0 measures it)
    segments: int = 25  # S: gate units per note (D3)
    flagged_fraction: float = 0.4  # phi: share of the text the gate passes on (D3)
    flagged_candidate_fraction: float = 0.6  # share of candidates inside flagged segments (PHI-shaped strings cluster)
    context_tokens: int = 130  # per-candidate snippet (+-200 chars) plus framing in the per-candidate designs
    note_prefix_candidate_tokens: int = 20  # per-candidate user turn when the whole note is in the cached prefix (D1n)
    marker_tokens: int = 6  # per-candidate overhead of inline markers [[12|...]] in the batched designs
    group_marker_tokens: int = 8  # per-occurrence marker with group and occurrence index (D6)
    decision_tokens: int = 6  # output per verdict in the compact line format, e.g. `12 DATE` / `12 -`
    positive_tokens: int = 5  # output per accepted candidate when only positives are listed (D2p)
    span_item_tokens: int = 15  # output per extracted span in F1 JSON
    tag_tokens: int = 12  # tokens added per inline tag pair in F3 / H5 output
    addition_fraction: float = 0.1  # D2b: free-text additions as a fraction of P
    token_index_overhead: float = 2.5  # D8: extra input tokens per note token for numbering every token
    phi_tokens_per_span: float = 2.5  # D8: tokens per PHI mention
    token_label_tokens: int = 5  # D8: output per labelled PHI token, e.g. `17 DATE`
    seconds_per_call: float = 1.0  # request overhead at concurrency 1 (rough; no prefill, no retries)
    output_tokens_per_second: float = 60.0  # decoding speed (rough; for wall-clock lower bounds)

    @property
    def k_fast(self) -> float:
        return self.fast_proposals if self.fast_proposals else round(self.overgeneration * self.phi_spans)


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. ``cache_read_mult`` is the cache-read input price as a multiple of the input price;
    ``batch_mult`` the batch-API multiplier applied to every token."""

    name: str
    input_per_m: float
    output_per_m: float
    cache_read_mult: float = 0.1
    batch_mult: float = 0.5


# Anthropic list prices as given by the bundled API reference (model table cached 2026-10-06, read 2026-11-02); cache
# reads ~0.1x input, batch 0.5x; the Haiku rate applies to prompts <= 100K tokens. Verify against the provider pricing
# page before quoting them in the paper, exactly as configs/litellm.models.yaml asks for the model identifiers.
PRICE_POINTS: dict[str, Prices] = {
    "sonnet-5": Prices("claude-sonnet-5", 2.00, 10.00),
    "opus-5": Prices("claude-opus-5", 5.00, 25.00),
    "haiku-5.5": Prices("claude-haiku-5-5", 0.10, 0.50),
    "local": Prices("local-vllm", 0.0, 0.0, 0.0, 1.0),  # GPU time is not modelled in USD
}


@dataclass(frozen=True)
class Part:
    """Token accounting of one model role inside a design, per note."""

    role: str  # "strong" | "cheap"
    calls: float
    input_tokens: float  # uncached input (note text, candidates, snippets)
    cached_tokens: float  # input served from the prompt cache (the system prompt on every call after the first)
    output_tokens: float

    def usd(self, p: Prices, cache: bool = True, batch: bool = False) -> float:
        cached_price = p.input_per_m * (p.cache_read_mult if cache else 1.0)
        tokens_usd = self.input_tokens * p.input_per_m + self.cached_tokens * cached_price
        usd = (tokens_usd + self.output_tokens * p.output_per_m) / 1e6
        return usd * (p.batch_mult if batch else 1.0)

    def seconds(self, w: Workload) -> float:
        return self.calls * w.seconds_per_call + self.output_tokens / w.output_tokens_per_second


@dataclass(frozen=True)
class Design:
    id: str
    name: str
    parts: Callable[[Workload], Sequence[Part]]
    local: str = "none"  # local compute outside the USD columns (proposer, TC pass, classifier, encoder)
    note: str = ""


@dataclass
class Estimate:
    design: Design
    parts: list[Part] = field(default_factory=list)

    @property
    def calls(self) -> float:
        return sum(p.calls for p in self.parts)

    @property
    def input_tokens(self) -> float:
        return sum(p.input_tokens for p in self.parts)

    @property
    def cached_tokens(self) -> float:
        return sum(p.cached_tokens for p in self.parts)

    @property
    def output_tokens(self) -> float:
        return sum(p.output_tokens for p in self.parts)

    def usd(self, prices: Mapping[str, Prices], cache: bool = True, batch: bool = False) -> float:
        return sum(p.usd(prices[p.role], cache, batch) for p in self.parts)

    def seconds(self, w: Workload) -> float:
        return sum(p.seconds(w) for p in self.parts)


# -- designs --------------------------------------------------------------------------------------------------------
def _f1(w: Workload) -> list[Part]:
    return [Part("strong", 1, w.note_tokens, w.guideline_tokens, w.phi_spans * w.span_item_tokens)]


def _f3(w: Workload) -> list[Part]:
    return [Part("strong", 1, w.note_tokens, w.guideline_tokens, w.note_tokens + w.phi_spans * w.tag_tokens)]


def _h5(w: Workload) -> list[Part]:
    tagged = w.note_tokens + w.phi_spans * w.tag_tokens
    return [Part("strong", 1, tagged, w.guideline_tokens, tagged)]


def _d1(w: Workload) -> list[Part]:
    k = w.candidates
    return [Part("strong", k, k * w.context_tokens, k * w.guideline_tokens, k * w.decision_tokens)]


def _d1n(w: Workload) -> list[Part]:
    k = w.candidates
    prefix = w.guideline_tokens + w.note_tokens
    return [Part("strong", k, k * w.note_prefix_candidate_tokens, k * prefix, k * w.decision_tokens)]


def _d2(w: Workload) -> list[Part]:
    k = w.candidates
    return [Part("strong", 1, w.note_tokens + k * w.marker_tokens, w.guideline_tokens, k * w.decision_tokens)]


def _d2p(w: Workload) -> list[Part]:
    (p,) = _d2(w)
    return [replace(p, output_tokens=w.phi_spans * w.positive_tokens)]


def _d2b(w: Workload) -> list[Part]:
    (p,) = _d2(w)
    return [replace(p, output_tokens=p.output_tokens + w.addition_fraction * w.phi_spans * w.span_item_tokens)]


def _d3(w: Workload) -> list[Part]:
    s, phi, k = w.segments, w.flagged_fraction, w.candidates
    kf = w.flagged_candidate_fraction * k
    gate = Part("cheap", 1, w.note_tokens + 4 * s, w.gate_tokens, 3 * phi * s)
    decide = Part("strong", 1, phi * w.note_tokens + kf * w.marker_tokens, w.guideline_tokens, kf * w.decision_tokens)
    return [gate, decide]


def _d3c(w: Workload) -> list[Part]:
    return _d3(w)[1:]  # the gate is a local classifier: no API cost


def _d4(w: Workload) -> list[Part]:
    k = w.candidates
    return [Part("strong", k, k * w.context_tokens, k * w.guideline_tokens, k * 1.0)]


def _d5(w: Workload) -> list[Part]:
    return []  # inference is a local encoder; training = D2 over the silver pool, one-off (see section 3.6)


def _d6(w: Workload) -> list[Part]:
    k = w.candidates
    e = k / w.repetition
    return [Part("strong", 1, w.note_tokens + k * w.group_marker_tokens, w.guideline_tokens, e * w.decision_tokens)]


def _d7(w: Workload) -> list[Part]:
    k2 = w.k_fast
    propose = Part("cheap", 1, w.note_tokens, w.guideline_tokens, w.overgeneration * w.phi_spans * w.span_item_tokens)
    decide = Part("strong", 1, w.note_tokens + k2 * w.marker_tokens, w.guideline_tokens, k2 * w.decision_tokens)
    return [propose, decide]


def _d8(w: Workload) -> list[Part]:
    phi_tokens = w.phi_spans * w.phi_tokens_per_span
    inp = w.note_tokens * (1 + w.token_index_overhead)
    return [Part("strong", 1, inp, w.guideline_tokens, phi_tokens * w.token_label_tokens)]


DESIGNS: tuple[Design, ...] = (
    Design("F1", "extraction, JSON strings (reference)", _f1),
    Design("F3", "extraction, inline tags", _f3),
    Design("H5", "TC-first editor, full tagged text out", _h5, "TC pass"),
    Design("D1", "one verdict call per candidate, snippet context", _d1, "proposer", "K calls/note; guideline cached"),
    Design("D1n", "one verdict call per candidate, note in cached prefix", _d1n, "proposer", "K calls/note"),
    Design("D2", "batched verdicts over marked candidates", _d2, "proposer", "output-bounded form of H5"),
    Design("D2p", "D2, positives-only output", _d2p, "proposer", "cheapest D2 output; re-admits omission"),
    Design("D2b", "D2 + free-text additions for unproposed PHI", _d2b, "proposer"),
    Design("D3", "cheap-model segment gate, then D2 on flagged", _d3, "proposer", "gate on a Haiku-class model"),
    Design("D3c", "classifier segment gate, then D2 on flagged", _d3c, "proposer + gate classifier", "gate is local"),
    Design("D4", "single-token answer scored by logprob (API variant)", _d4, "proposer", "needs logprobs"),
    Design("D5", "distilled decision classifier (inference)", _d5, "proposer + encoder", "training = one D2 pass"),
    Design("D6", "D2 with one verdict per distinct string", _d6, "proposer", "saving = 1 - 1/rep"),
    Design("D7", "cheap over-generation, then strong D2", _d7, "none", "proposer on a Haiku-class model"),
    Design("D8", "numbered-token labelling (proposer-free control)", _d8, "none", "R_C = 1 by construction"),
)
DESIGN_BY_ID: dict[str, Design] = {d.id: d for d in DESIGNS}
DEFAULT_WORKLOAD = Workload()


def estimate(design: Design | str, w: Workload = DEFAULT_WORKLOAD) -> Estimate:
    d = DESIGN_BY_ID[design] if isinstance(design, str) else design
    return Estimate(d, list(d.parts(w)))


def estimate_all(w: Workload = DEFAULT_WORKLOAD, designs: Sequence[Design] = DESIGNS) -> list[Estimate]:
    return [estimate(d, w) for d in designs]


# -- table ----------------------------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 or x == 0 else f"{x:.1f}"


def markdown_table(w: Workload, prices: Mapping[str, Prices], notes: int = 1000, reference: str = "F1") -> str:
    """Per-note tokens and USD per ``notes`` notes (prompt cache on; batch column = cache + batch API). The hours
    column is a lower bound at concurrency 1 (request overhead + decoding; no prefill, no retries)."""
    ests = estimate_all(w)
    ref = estimate(reference, w).usd(prices, cache=True)
    head = (
        f"| ID | Design | local compute | calls/note | input (uncached) | input (cache read) | output | "
        f"USD/{notes:,} no cache | USD/{notes:,} cache | x{reference} | USD/{notes:,} cache+batch | "
        f"h/{notes:,} @1 (lb) |"
    )
    rows = [head, "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in ests:
        nocache, cache, batch = (e.usd(prices, c, b) for c, b in ((False, False), (True, False), (True, True)))
        ratio = f"{cache / ref:.2f}" if ref else "n/a"
        hours = e.seconds(w) * notes / 3600
        rows.append(
            f"| {e.design.id} | {e.design.name} | {e.design.local} | {_fmt(e.calls)} | {_fmt(e.input_tokens)} | "
            f"{_fmt(e.cached_tokens)} | {_fmt(e.output_tokens)} | {nocache * notes:,.2f} | {cache * notes:,.2f} | "
            f"{ratio} | {batch * notes:,.2f} | {hours:.1f} |"
        )
    return "\n".join(rows)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for f in Workload.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    ap.add_argument("--strong", default="sonnet-5", choices=sorted(PRICE_POINTS))
    ap.add_argument("--cheap", default="haiku-5.5", choices=sorted(PRICE_POINTS), help="gate (D3) / proposer (D7)")
    ap.add_argument("--input-per-m", type=float, help="override the strong role input price (USD per 1M tokens)")
    ap.add_argument("--output-per-m", type=float, help="override the strong role output price (USD per 1M tokens)")
    ap.add_argument("--cache-read-mult", type=float, help="override the strong role cache-read multiplier")
    ap.add_argument("--notes", type=int, default=1000, help="number of notes the USD columns are quoted for")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> str:
    a = _parse(argv)
    w = Workload(**{k: getattr(a, k) for k in Workload.__dataclass_fields__})
    strong = PRICE_POINTS[a.strong]
    if a.input_per_m is not None or a.output_per_m is not None or a.cache_read_mult is not None:
        strong = Prices(
            "custom",
            a.input_per_m if a.input_per_m is not None else strong.input_per_m,
            a.output_per_m if a.output_per_m is not None else strong.output_per_m,
            a.cache_read_mult if a.cache_read_mult is not None else strong.cache_read_mult,
            strong.batch_mult,
        )
    prices = {"strong": strong, "cheap": PRICE_POINTS[a.cheap]}
    header = (
        f"Workload: N={w.note_tokens} G={w.guideline_tokens} P={w.phi_spans} K={w.candidates} K'={w.k_fast} "
        f"S={w.segments} phi={w.flagged_fraction} phi_K={w.flagged_candidate_fraction} rep={w.repetition} "
        f"verdict={w.decision_tokens}; strong={strong.name} ({strong.input_per_m}/{strong.output_per_m} USD per 1M, "
        f"cache read x{strong.cache_read_mult}), cheap={prices['cheap'].name} "
        f"({prices['cheap'].input_per_m}/{prices['cheap'].output_per_m})"
    )
    out = header + "\n\n" + markdown_table(w, prices, a.notes)
    print(out)
    return out


if __name__ == "__main__":
    main()
