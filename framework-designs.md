# Framework designs: hybrid LLM + token-classifier de-identification of clinical notes

Status: **draft for design review** (2026-09-14; §3.6 decision-model family and Appendix D added 2026-11-02). Items marked ▶ are decisions still open; each carries a recommended
default. Everything else is proposed as the working design. Literature references are keyed to `literature.md`.

---

## 0. One-paragraph summary

We have two working de-identification systems with different failure modes: a guideline-prompted LLM that returns spans as
JSON and *misses* spans (and mis-localizes some), and a fine-tuned token classifier (TC; OpenAI Privacy Filter or similar)
that is fast and precise on frequent labels but *cannot learn rare labels* and *does not exist without annotated data*. The
proposal is to (1) measure how complementary the two error sets are, per label and per training frequency, (2) evaluate a
structured space of hybrids — inference-time fusion, cascades, training-time coupling, LLM grounding, surrogation — against
both single systems under a recall-first protocol, (3) do this across a controlled range of human-label budgets, and (4)
release a reproducible PHI-injection benchmark with controllable label frequencies so the rare-label regime can be studied
even while gated corpora are unavailable. The paper's claim is not "ensembles help" (homogeneous LLM ensembles do not:
Palacios 2026) but that **heterogeneous systems with structurally different failure modes can be combined by routing on the
structure of those failures**, and that the LLM can *manufacture* the training signal the TC lacks. A fourth axis,
decision-model detection (family D, §3.6), asks whether the LLM should *judge explicit candidates* instead of
*enumerating spans*: that removes grounding failures by construction and bounds the output, at the price of a
candidate-coverage ceiling and a cost profile that Appendix D compares design by design.

---

## 1. Problem statement and scope

### 1.1 Task, input/output, target taxonomy

- **Input**: one clinical note (free text, any note type). **Output**: a set of spans `(start, end, label)` over the
  original character offsets. Downstream: redaction (`[LABEL]` placeholder) or surrogation (realistic replacement).
- **Canonical taxonomy**: the i2b2/UTHealth 2014 scheme — 7 categories, 28 leaf labels: 25 sub-categories under NAME, LOCATION, CONTACT, ID plus the undivided PROFESSION, AGE, DATE (see `src/note_deid/labels.py`) —
  because it is the community benchmark, it is a superset of HIPAA Safe Harbor, and it includes quasi-identifiers
  (PROFESSION, AGE < 90, HOSPITAL, all DATEs) that the user's guideline already covers. Evaluation is reported at three
  granularities: 28 leaf labels, 7 categories, HIPAA-only subset.
- ▶ **Institution-specific extension** (recommended: yes, behind a flag). Add a label group `INSTITUTIONAL` (internal
  department/clinic codes, building names, local abbreviations) realized in the injection benchmark through a configurable
  institution lexicon. Motivation: Palacios 2026 shows this is where purpose-built systems fail and gold standards are
  incomplete; it is also the sharpest instance of "no training data for this label".
- **Quasi-identifier policy**: the guideline (prompt) is the single source of truth for what counts; the TC inherits it
  through its training labels. Contextual decisions with low human agreement (RedactionBench: 47.7%) are tracked as a
  separate "contextual" subset in evaluation rather than argued away.

### 1.2 The two base systems as they exist today (to be imported from prior work — Appendix A)

| | S_LLM (guideline + LLM) | S_TC (fine-tuned token classifier) |
|---|---|---|
| Model | commercial API model via LiteLLM | OPF (1.5B MoE, 8 labels) fine-tuned; alternatives DeBERTa-v3, OpenMed-PII |
| Supervision | annotation guideline text, expressed as a DSPy program and refined with GEPA (reflective prompt evolution) against annotated notes | BIO/BIOES labels from annotated notes |
| Output | JSON list of `{text, label}` → exact string search → spans | token logits → BIOES decoding → spans with probabilities |
| Determinism | no (even at T=0) | yes |
| Cost / note | ~1–3k input tokens + output; seconds | milliseconds; negligible |
| Label set | anything the guideline describes | fixed at training time |

### 1.3 Failure-mode table (observed + literature)

| Failure mode | S_LLM | S_TC | Evidence |
|---|---|---|---|
| Rare label (few training instances) | robust (label defined in guideline) | **weak / absent** | SHIELD macro-recall gap; i2b2 long tail; our experiments |
| Label not in taxonomy / policy change | prompt edit | **retrain required** ("static label policy") | OPF README |
| No annotated data at all | works zero-shot | **no model** | — |
| Long-tail surface formats (letterheads, fax lines, odd emails) | usually fine | **misses** | Tonic.ai OPF benchmark |
| Institution-specific identifiers | good if prompt names them | poor without local data | Palacios 2026; locally augmented ensembles |
| Context-dependent PHI (needs reasoning) | good | weak | Palacios: TiDE Other-ID recall 15.7% |
| Omitted mentions in long notes / repeated entities | **misses** (the user's core observation) | fine | Altalla' 2025 recall 0.83; our experiments |
| Span grounding (non-verbatim, ambiguous repeats, offsets) | **fails** with JSON+search | exact by construction | span-labeling format studies |
| Over-redaction / clinical information loss | **yes**, correlated across runs | rare | Palacios; EMNLP 2025 survey |
| Precision on frequent labels | good–very good | **very good** | Altalla' P 0.99; OPF precision comparable |
| Non-determinism | yes | no | Palacios ±0.002 F1 |
| Cost, latency, throughput | high | negligible | — |
| Boundary conventions (titles, suffixes, date ranges) | drifts | learned | i2b2 strict vs relaxed gap |

**Reading of the table.** The two systems fail on nearly disjoint *rows*. That is the structural reason to expect
heterogeneous complementarity where homogeneous LLM ensembles show none.

### 1.4 Thesis and what would falsify it

- **T1 (complementarity)**: for gold spans, the miss events of S_LLM and S_TC are weakly dependent; conditional miss
  probability P(TC misses | LLM misses) is much lower than P(TC misses | LLM catches) only if errors are correlated —
  we predict the *oracle union* recall exceeds the best single system by a margin concentrated in rare labels (TC misses) and
  in repeated/long-note mentions (LLM misses).
- **T2 (fusion)**: a hybrid that routes on failure structure (label frequency, TC confidence, LLM grounding) reaches recall
  within 1 point of the oracle union while keeping precision within 2 points of the better single system, at a fraction of
  the LLM-on-everything cost.
- **T3 (label scarcity)**: with ≤100 human-labeled notes, LLM-manufactured supervision (silver labels + rare-label injection +
  disagreement-driven annotation) brings the TC — and therefore the hybrid — within a few F1 points of full supervision.
- **Falsification**: if the FN-set Jaccard between systems exceeds ~0.5 and oracle-union recall is < 2 points above the best
  single system, T1 fails and the study becomes an analysis paper on why (still reportable). If fusion cannot recover
  precision (union-level FP rate persists), T2 fails and the recommendation becomes "TC + LLM-verified rare labels only".

---

## 2. Formal setup and notation

- Document x with gold spans G(x) = {g = (s, e, ℓ)}. System k ∈ {LLM, TC, …} outputs Y_k(x) = {(s, e, ℓ, score, source)}.
- Matching m(y, g): **strict** = identical (s, e, ℓ); **relaxed** = character overlap and same ℓ; **type-agnostic relaxed** =
  overlap only (privacy view: a date masked as NAME is still masked).
- FN_k(x) = {g ∈ G(x) : no y ∈ Y_k(x) matches g}; FP_k(x) analogously.
- Per-label training frequency n_ℓ (count of gold spans of subtype ℓ in the TC training split); frequency buckets
  B = {0, 1–10, 11–50, 51–200, > 200}.
- Complementarity statistics: J = |FN_LLM ∩ FN_TC| / |FN_LLM ∪ FN_TC|; conditional miss P(g ∈ FN_TC | g ∈ FN_LLM) vs marginal
  P(g ∈ FN_TC); **oracle union recall** R_∪ = 1 − |FN_LLM ∩ FN_TC| / |G|; oracle-intersection precision; all reported per label,
  per bucket, per note type, per mention form (full name / partial / initials / possessive).
- Cost c_k(x): LLM = input tokens × price_in + output tokens × price_out; TC = GPU-seconds. Hybrid cost = sum over calls made.
- A **hybrid** is a pair (routing policy π, fusion function f): π decides which systems run on which units (note, segment,
  label), f maps their outputs to the final span set.

---

## 3. Design space

Each design lists: mechanism · failure modes targeted · requirements · cost · risks · expected outcome · priority
(P1 = core of the paper, P2 = second wave, P3 = optional).

### 3.1 Inference-time fusion (both systems run; outputs combined)

**H1 — Union (recall ceiling).** Y = merge(Y_LLM ∪ Y_TC); overlapping spans with the same label are merged to the union
extent; conflicting labels resolved by a priority rule (TC label if TC span score ≥ θ, else LLM label; configurable).
Targets: every miss of either system. Requires: nothing. Cost: LLM + TC. Risk: precision = combined false positives.
Expected: recall ≈ R_∪; precision drop equal to the FP union. Serves as the upper reference for all other fusions. **P1**

**H2 — Label-routed fusion.** For each label ℓ choose the source set: TC-only if n_ℓ ≥ τ (or dev-F1_TC(ℓ) ≥ dev-F1_LLM(ℓ)),
LLM-only if ℓ is unsupported by the TC or n_ℓ < τ, both (union) for "critical" labels (configurable, e.g. NAME, ID). Sweep τ.
Targets: TC rare-label misses without importing LLM noise on frequent labels. Requires: label frequency table (always
available) or a small dev set. Cost: LLM + TC (LLM output can be restricted to the routed labels to save tokens). Risk: a
label's frequency is a proxy; some frequent labels may still be format-fragile. Expected: keeps TC precision on the head,
LLM recall on the tail; the simplest hybrid likely to beat both singles. **P1**

**H3 — Confidence-gated arbitration.** TC span score = min (or mean) token probability of the decoded span; LLM span carries
a grounding score (exact / fuzzy / ambiguous). Rules: (a) both systems agree → accept; (b) TC-only with score ≥ θ_hi →
accept; (c) TC-only with score < θ_hi → accept only if a cheap LLM verification call (span + ±200 chars, yes/no) confirms,
else drop; (d) LLM-only → accept if ℓ is TC-unsupported or rare, else accept only if TC forced-decoding probability for ℓ
over that extent ≥ θ_lo or a verification call confirms. θ tuned on dev for a target recall. Targets: union-level FP while
retaining union-level recall. Requires: TC probabilities; small dev set for thresholds. Cost: LLM + TC + verification calls
(short). Risk: verification calls share LLM biases (Palacios: self-verification found nothing) — mitigated because here the
verifier checks the *other* system's output. Expected: the best precision–recall trade-off among unlearned fusions. **P1**

**H4 — Learned stacking (meta-combiner).** Candidate set C = unique extents from Y_LLM ∪ Y_TC; features per candidate: TC
span probability and label margin (0 if absent), LLM presence and label, agreement flag, label one-hot, span length,
character-class pattern (digits / caps / mixed), position class (header / body / signature block), in-document frequency of
the string, note type, regex-cue flags (date, phone, email, id shapes). Model: logistic regression or gradient boosting with
per-label calibration; decision threshold chosen on dev for a target recall. Targets: all of H3 without hand-tuned rules.
Requires: 100–300 labeled notes for the combiner (evaluated on the learning curve). Cost: LLM + TC. Risk: overfits to the
benchmark's injection artifacts — validate on a second source. Expected: best Pareto frontier when labels exist. **P2**

### 3.2 Cascades (one system conditions the other)

**H5 — TC-first, LLM-as-editor (grounded review).** Run TC; render its spans inline
(`<PHI type="DATE">03/04/2019</PHI>`); prompt the LLM with the guideline to *review*: add missing tags, remove wrong tags,
fix types and boundaries, and return the full tagged text with all other characters unchanged (tagging family) — or return an
edit list keyed by occurrence index. Parse by diff-alignment against the original; if non-tag characters changed, reject the
output and fall back to H1. Variant H5b: the prompt names the labels the TC cannot produce, so the LLM concentrates on them.
Targets: LLM omissions (reviewing a pre-tagged text is a recognition task, not an enumeration task), LLM grounding failures
(tags are anchored in the text), TC rare-label and format misses (LLM adds), TC false positives (LLM removes). Requires:
nothing beyond both systems. Cost: LLM input ≈ note + tags, output ≈ note (tagging) or small (edit list). Risks: anchoring —
the LLM may over-trust TC tags (measure removal rate of TC FPs); output-length cost on long notes (mitigate with
segment-level editing). Expected: the cascade with the largest recall gain per LLM token; the key design for the paper's
"LLM misses" narrative. **P1**

**H6 — LLM-first, TC safety net.** LLM runs first; TC adds spans only for its high-precision labels. Mostly subsumed by H2;
kept for the cost analysis when the LLM is mandatory for policy reasons. **P3**

**H7 — Selective LLM invocation (cost-aware routing).** Run TC on every note; compute a risk score r(x) from: TC uncertainty
mass (tokens whose max probability lies in [0.2, 0.8]), count of regex cues for rare labels not covered by TC spans (emails,
URLs, fax/phone shapes, id-like tokens), note length, presence of header/footer/signature blocks, note type, count of
capitalized out-of-vocabulary tokens. Send only notes (or only flagged segments) with r(x) > ρ to the LLM (from-scratch or
H5 editor). Sweep ρ → recall-vs-cost curve. Targets: cost; keeps most of the union gain if misses concentrate in flagged
units. Requires: TC probabilities. Risk: silent misses in unflagged notes — report the recall of unflagged notes separately.
Expected: a Pareto curve between TC-only and LLM-on-everything; the operational recommendation of the paper. **P1**

**H8 — Residual-PHI audit loop (heterogeneous verifier).** After redaction by the TC (or by a hybrid), the LLM inspects the
redacted text and lists any remaining identifiers; iterate at most twice. Palacios found LLM self-audits of LLM output add
nothing; the open question is whether an LLM audit of *TC* output does. Cheap to test. **P2**

### 3.3 Training-time coupling (the LLM manufactures supervision for the TC)

**H9 — LLM-silver distillation.** Label an unlabeled pool with S_LLM (best-grounded format from H15); optional
self-consistency: k = 3 samples at T > 0, keep spans with ≥ 2 votes (precision-oriented) or ≥ 1 (recall-oriented) — evaluate
both; train the TC on silver labels only; compare to a TC trained on the same number of *human*-labeled notes and to S_LLM
itself. Iterated variant: TC₀ (silver) pre-tags → H5 editor produces silver₁ → TC₁ … (self-training with a heterogeneous
teacher). Targets: the "no annotated data → no model" failure. Requires: unlabeled notes only (test set excepted). Cost:
one LLM pass per silver note (one-off). Risks: silver noise on boundaries; LLM omissions become TC blind spots — which is
why iteration through H5 matters. Expected: silver TC ≈ human TC on frequent labels; tail depends on LLM coverage. **P1**

**H10 — Rare-label augmentation by controlled PHI injection.** Into the *training* notes only, inject rare-label PHI at
controlled counts (0 / 10 / 50 / 200 per label) using (a) template insertion into realistic slots (headers, signature
blocks, "cc:" lines), (b) LLM contextual insertion (the LLM rewrites a sentence to mention the given entity naturally and
returns the tagged sentence; the entity string must appear verbatim, so gold offsets are exact by construction). Also
*counterfactual* augmentation: swap existing entity strings for diverse surrogates (uncommon names, international formats)
to attack the TC's uncommon-name weakness. Targets: TC rare-label recall and format brittleness. Requires: a base training
set (human or silver) and the injection engine (shared with the benchmark). Cost: generation only. Risks: unrealistic
injection → false positives on real-distribution text; measure FP on an untouched test split and on a second source.
Expected: monotone rare-label recall gains with diminishing returns after ~50 instances per label. **P1**

**H11 — Disagreement-driven active learning.** Simulate annotation rounds with budget b notes/round from a pool: strategies
= random · TC uncertainty · **LLM–TC disagreement** (size of the symmetric difference of span sets, up-weighting rare labels) ·
disagreement + diversity; the "annotator" returns gold labels. Metric: test F1 (overall and rare labels) vs total labeled
notes; area under the learning curve. Targets: label efficiency, especially for the tail. Requires: gold labels for the pool
(available in the benchmark). Expected: disagreement sampling dominates on rare labels because disagreements concentrate
exactly where one system fails. **P2**

**H12 — Guideline ↔ model co-refinement loop** (formalizes the existing guideline-refinement practice, in which the guideline is a DSPy program optimized with GEPA; G₀ is that optimized guideline and the GEPA optimizer can serve as the edit-proposal mechanism in Step A). State (G_k, TC_k).
Step A (LLM side): run S_LLM(G_k) on dev₁; an LLM critic proposes guideline edits (added decision rules, examples,
clarified boundaries) from the FN/FP list; edits are accepted by a human (or automatically, as an ablation) → G_{k+1};
overfitting guard: improvement must replicate on dev₂ before acceptance. Step B (TC side): categorize TC_k errors on dev₁;
rare-label misses trigger targeted H10 injection, format misses trigger format-variant augmentation, then retrain →
TC_{k+1}. Step C (coupling): G_{k+1} re-labels the silver pool (H9), and TC_{k+1}'s confident spans feed the H5 editor, so
each side's improvement propagates to the other. Track per-label curves over k. Targets: both systems' residual errors;
documents which errors move from "one system catches" to "both catch". Requires: ~100–200 annotated notes split into
dev₁/dev₂ (the prior annotated notes). Expected: convergence in 2–3 rounds. **P2**

**H13 — Label-set extension of the OPF head.** Options: (a) replace the 33-class head with a 28-label BIOES head,
initializing rows for mapped labels from OPF's rows (PATIENT/DOCTOR ← private_person, DATE ← private_date, STREET/CITY/…
← private_address, PHONE/FAX ← private_phone, EMAIL ← private_email, URL ← private_url, ID* ← account_number) and new rows
(AGE, PROFESSION, HOSPITAL, ORGANIZATION, USERNAME, IPADDR, COUNTRY) at random; fine-tune fully or with LoRA; (b) keep the
OPF head and add a second small head for the unsupported labels; (c) DeBERTa-v3-base/large trained from scratch on the
same data; (d) OpenMed-PII (54 labels) mapped and fine-tuned. Data-efficiency curve at 25 / 50 / 100 / 200 / 400 notes.
Targets: the fixed-taxonomy failure and the data-efficiency claim of privacy-pretrained encoders. Requires: labeled or
silver notes. Expected: OPF most data-efficient on mapped labels; new labels need ≳ 50–100 instances, which H2 routing to
the LLM covers until then. **P1**

### 3.4 Representation-level coupling and LLM grounding

**H14 — LLM-hint-conditioned TC.** Feed LLM tags as special marker tokens (or an extra feature channel) to the TC at training
and inference so it learns when to trust the hint. Requires the LLM on every note (same cost as H5) and silver-tagged training
data. Risk: train/inference mismatch. **P3**

**H15 — LLM output-format / grounding ablation** (explains the "LLM misses" before hybrids are judged). Formats:
F1 JSON list of strings + exact search (current pipeline); F2 JSON with a left-context anchor or occurrence index +
anchored search; F3 inline tagging of the full text, validated by diff (reject altered text); F4 inline tagging per
numbered sentence/segment (bounds output length); F5 constrained decoding on a local model (vLLM + grammar / LogitMatch-style)
— optional. Alignment fallbacks: whitespace/punctuation normalization, fuzzy match (partial ratio ≥ 0.9), expansion to all
occurrences. Measures: parse-failure rate, alignment-failure rate, recall/precision, output tokens. Decomposition:
misses = parse failures + alignment failures + semantic omissions. Expected: F3/F4 remove most grounding failures at higher
output cost; the semantic-omission residual is what hybrids must address. **P1**

**H16 — Zero-shot span model as a third component.** GLiNER-PII / OpenMed-PII with natural-language label names: no
training, cheap, extensible labels. Uses: extra union member; gate features for H7; source for TC-unsupported labels before
H13 has data. **P3**

### 3.5 Surrogation (replacing rather than redacting)

**S1 — Rule/Faker surrogates.** Type-consistent generators; document-level consistent mapping (same string → same surrogate,
name variants handled by name parsing); per-patient date shift preserving intervals and weekdays; ages > 89 → "90+"; ID
shapes preserved by regex-shape-preserving generation; ZIP → 3-digit rule. **S2 — LLM surrogates.** Given span, label,
context and constraints, the LLM generates a plausible, format-preserving replacement; post-checks: not equal to any real
PHI string in the note, detectable by the TC (to preserve the hiding-in-plain-sight effect). **S3 — Hybrid.** Rules for
structured types (DATE, ID*, PHONE/FAX, ZIP, AGE), LLM for NAME variants, HOSPITAL/ORGANIZATION, LOCATION-OTHER,
PROFESSION, free-form mentions. Detection input for all three is the best hybrid detector.
Metrics: residual leakage (count; attacker test — a classifier/LLM tries to separate real leaks from surrogates; report
attacker precision on the residual leaks, cf. Carrell 2013/2020), mapping-consistency rate, realism (human/LLM judge),
detector re-run equivalence (TC recall on surrogated vs original within ±2 points, TOST), utility (downstream note-type
classification and clinical-concept NER drift; readability). **P2** (design now, run in phase P4).

### 3.6 Decision-model detection (family D: the LLM judges candidates instead of enumerating spans)

S_LLM in §1.2 is an *extraction* model: it reads the note and enumerates PHI spans (formats F1–F4). A **decision model**
answers a closed question about a unit it is handed — *is this candidate PHI, and which label?* — so the generative part of
the task shrinks to a verdict, offsets are known by construction (the candidate carries them), and output length is
bounded by the number of candidates rather than by the note. The H3 verifier (Appendix B) is already a decision model,
applied only to disputed candidates; the H5 editor is a decision model in disguise (a verdict per pre-tagged span plus
additions) that pays for its verdicts by re-emitting the whole note. Family D makes the decision model the primary LLM
role. In the project's earlier notes this is "model 1 (decision)", as opposed to the extraction model.

**Why it may fix what extraction cannot.** The LLM failures of §1.3 split into (i) *omission* — the model never emits a
span that is in the text — and (ii) *grounding* — it emits a non-verbatim string or an ambiguous repeat. Decision
prompting removes (ii) entirely and turns (i) from recall-from-memory (enumerate everything you noticed) into recognition
(answer for each marked item) — the argument behind H5, now with a bounded output. The price is a new failure mode,
**candidate miss**: a gold mention that no proposer offers can never be recovered. Every design below is therefore judged
on three numbers — candidate coverage R_C (fraction of gold spans with a matching candidate, strict and relaxed), decision
quality (precision/recall over candidates), and cost per note — and end-to-end misses decompose as candidate misses +
decision misses + boundary mismatches, extending the RQ8 decomposition.

**Design axes.**

- *Unit of decision*: candidate span · distinct string (entity) · segment/sentence · token.
- *Candidate proposer*: (a) **surface-form** — regex shapes (dates, phones, fax, emails, URLs, IPs, SSN/ID shapes, ZIPs,
  ages), capitalized token runs not at sentence start, titled names (Dr./Mr./Ms.), digit runs, values on header lines
  (`HEADER_LINE` in `hybrid/routing.py`); recall-maximizing, ≈ 60–100 candidates per 1k-token note; (b) **TC-based** — TC
  spans plus uncovered regex cues (`RARE_CUE_PATTERNS`) plus low-confidence tokens, ≈ 30; (c) **zero-shot span model**
  (GLiNER, H16); (d) **cheap LLM over-generation** (an extraction pass told to list anything that could be PHI); (e) the
  union of (a)–(d). Coverage is measured per proposer before any decision design runs (D0).
- *Decision function*: strong LLM with JSON verdicts · LLM with a one-token answer scored by log-probabilities
  (calibrated) · distilled encoder over the marked context · local model with prefix caching.
- *Context*: ±200-character snippet vs the whole note (document-level cues: a bare first name is PHI because the full name
  is in the header; letterheads; signature blocks).
- *Batching*: one call per candidate · one call per note · one call per flagged segment.

**D0 — Candidate-coverage analysis (no LLM).** For each proposer (a)–(e): R_C strict/relaxed per label and per frequency
bucket against gold on dev, and the candidate-to-gold ratio (the noise a decision model must absorb). Fixes the recall
ceiling of every D design and the K that drives their cost. Requires: a proposer module (`llm/candidates.py`, planned)
sharing its regexes with H7. Expected: relaxed R_C ≥ 0.98 for the union proposer at K ≈ 80–120; strict R_C lower on
multi-token names and suffixed dates unless candidates are maximal runs; gaps concentrate in lowercase names, PROFESSION,
LOCATION-OTHER and HOSPITAL abbreviations. **P1** (cheap; prerequisite for the family)

**D1 — One decision call per candidate (the H3 verifier generalized).** For every candidate: system = guideline + verdict
instruction (a cacheable prefix), user = snippet with the candidate bracketed → `{"is_phi": bool, "label": ...}`.
Targets: omission and grounding; isolates the decision function for calibration studies. Requires: the existing
`build_verifier_messages`. Cost: K calls per note; with the guideline served from the prompt cache each call is
≈ 0.1 G + 130 input-price-equivalent tokens (K × G without caching) — ≈ 10 × F1 and ≈ 80 requests per note, i.e. about
26 h per 1k notes at concurrency 1 (Appendix D). Risks: the snippet loses document-level cues; per-request rate limits
dominate; cache entries expire when calls are spaced more than 5 min apart. Expected: the most accurate single-candidate
verdict (no attention dilution) but unaffordable as a primary detector; its roles are the reference decision function
and the H3 verifier. **P2**

**D2 — Batched decisions over marked candidates (one call per note).** Numbered markers are inserted around every
candidate (`⟦12|03/04/2019⟧`), the guideline stays the system prompt, and the model returns one verdict per index:
`{"decisions": [{"i": 12, "phi": true, "label": "DATE"}, ...]}`; offsets come from the marker table, never from the
output. Variants: **D2p** positives-only output (`12:DATE` per line) — the cheapest, but it re-admits omission because an
unmentioned candidate silently defaults to "not PHI" (measure the gap to D2); **D2b** verdicts plus a free-text
`"additions"` list for PHI no marker covers (an enumeration channel for the residual, aligned like F1) — closes the
candidate ceiling at small cost; **boundary edits** — a verdict may grow or trim a candidate by whole tokens
(`"extend": [1, 0]`) so maximal-run candidates can be corrected. Targets: omission (a recognition task over every
candidate), grounding (by construction), H5's output cost (verdicts instead of the full note). Requires: proposer, marker
renderer and parser; a `decision` detector type in the runner. Cost: one call; input = note + ≈ 6 tokens per marker;
output ≈ 10 tokens per candidate (D2) or ≈ 5 per positive (D2p) → ≈ 2.1 × F1 (D2) and ≈ 0.8 × F1 (D2p) at the default
workload; ≈ 4 h per 1k notes at concurrency 1. Risks: marker clutter on long notes (sweep K; drop rule-negative candidates
such as section headers and lexicon drug names); attention dilution above K ≈ 150 (segment as in F4); verdict-list
drift — missing or duplicated indices are counted as parse failures. Expected: recall ≥ F3 at 0.8–2.1 × F1 cost; the
structured-output form of H5 and the **primary design of the family**. **P1**

**D3 — Segment gate, then decisions on flagged segments (two-stage decision cascade).** Stage 1 decides per segment
(line / paragraph / sentence) whether it contains PHI and which labels: a cheap LLM over the numbered segments (short
system prompt; output = flagged indices) or, **D3c**, a small sentence classifier trained on silver verdicts. Stage 2 runs
D2 over the flagged segments only (φ ≈ 0.3–0.5 of the text where PHI concentrates in headers, signature blocks and a few
sentences). Targets: strong-model cost; routing by content rather than by TC uncertainty (complements H7 and works with
no TC at all). Requires: `segment_text`; gate prompt or classifier. Cost: ≈ 0.9 × F1 with the gate on the fast alias.
Risks: a gate miss is a silent miss (report recall on unflagged segments, as for H7); φ is corpus-dependent — on dense
notes the gate saves nothing. Expected: a recall–cost curve over the gate threshold that dominates H7 when TC uncertainty
is uninformative. **P2**

**D4 — Token-level decision by log-probability scoring (calibrated decision model).** The D1 prompt, but the answer is a
single token (`yes`/`no`, or a label letter) and the decision score is its log-probability; the threshold is set on dev
for a target recall (the recall-first protocol) instead of trusting a sampled verdict. API variant: needs `logprobs`
(OpenAI-compatible endpoints, local vLLM; not exposed by the Anthropic API) and costs like D1 with one output token.
Local variant: with prefix caching the note is prefilled once and each candidate is a short continuation, so the cost is
GPU-seconds (≈ G + N + 40 K prefill tokens per note) rather than USD — the natural deployment of the family on the A100
node (`llm-local`). Targets: calibration (ECE, risk–coverage curves); recall-first thresholding; a score that H3's
`verify` and H4's features can consume. Requires: `logprobs` support in `LLMClient`; a local model. Risks: verbalized
yes/no probabilities of instruction-tuned models are over-confident (temperature scaling or isotonic regression on dev);
answer ordering (label-first vs yes/no-first) changes the scores. Expected: a tunable recall–precision curve where D1/D2
give a single point. **P2**

**D5 — Distilled decision classifier (cross-encoder).** Train a small encoder (DeBERTa-v3-base) on D2 verdicts over the
silver pool (H9's unlabeled notes): input = snippet with the candidate marked plus the label *definition* text, output =
label or none. Because the definition is an input, a new label needs new text plus a handful of silver verdicts
(GLiNER-style extensibility) rather than a new head (H13). Targets: inference cost of the whole family (≈ 0 USD,
milliseconds per note); the no-data regime (silver verdicts are cheaper and more precise than silver spans because the
candidate is fixed). Requires: proposer; one D2 pass over the pool (one-off ≈ D2 cost × pool size); GPU. Risks: inherits
D2's omissions on the pool (iterate through D2b); a proposer change is a distribution shift. Expected: the decision
analogue of H9 — matches D2 on frequent labels, trails on the tail until silver verdicts cover it. **P3** (P2 if H9 proves
out)

**D6 — Entity-level decisions (one verdict per distinct string).** Group candidate occurrences by normalized string
(case, whitespace, punctuation; name variants by the name parsing of S1), render every occurrence with its group index,
and ask for one verdict per group with an explicit exception channel (`"except": [occurrence indices]`) for strings whose
role varies by context (`May` the month vs the name; `Jordan` the country vs the person). Targets: the "omitted repeated
mention" failure directly (one decision propagates to every occurrence); output cost by the repetition factor
(≈ 1.6 → 1.56 × F1 vs 2.12 × for D2). Requires: D2 plus grouping. Risks: wrong propagation on role-ambiguous strings
(measure exception usage and errors); group-index clutter. Expected: no recall loss vs D2 at ≈ 27 % lower cost (37 % fewer output tokens); the
operational form of D2. **P1** (as a D2 variant)

**D7 — Over-generate, then decide (two-model cascade).** A cheap model (or GLiNER, H16) runs an extraction pass with a
recall-maximizing instruction ("list anything that could be PHI; precision does not matter"), producing K' ≈ 1.5 P
candidates; the strong model runs D2 over them. Targets: K (the strong model reads ≈ 30 markers instead of 80), and the
precision role that Palacios found LLM self-verification cannot fill — here proposer and judge differ in model and in
instruction, so the judge's job is precision over an intentionally noisy set. Requires: two aliases; the proposer's R_C
from D0. Cost: ≈ 1.1 × F1. Risks: the proposer's omissions are unrecoverable (the ceiling is the cheap model's recall;
a union with the surface-form proposer restores it at higher K); correlated errors when proposer and judge share a
provider. Expected: the best cost–recall point when a surface-form proposer is too noisy for the note type
(narrative-heavy notes with many capitalized clinical terms). **P2**

**Relationship to the hybrids.** D2/D6 are drop-in replacements for the LLM in H1–H4 (a verdict set is a span set with
scores) and for the editor in H5; proposer (b) turns D2 into "TC-first, LLM decides" — the cascade H5 reaches for, with a
bounded output; D4 scores feed H3's verifier and H4's features; D3's gate is an alternative router to H7. **What the
family cannot do**: it does not help where the proposer has no candidate — lowercase or unusual-script names, free-text
PROFESSION, HOSPITAL abbreviations, LOCATION-OTHER — exactly the rows where the extraction LLM is strong (§1.3). D2b's
additions channel and a union with a cheap extraction run are the mitigations; E9 measures whether the residual is worth
them.

**Cost comparison.** Appendix D gives the closed-form per-note token and cost model (`note_deid.llm.costmodel`) for
F1/F3/H5 and D1–D7 under stated assumptions (1k-token note, 1.5k-token guideline, 20 PHI mentions, 80 candidates). At the
default workload and the Sonnet-5 price point: D1 ≈ 10 × F1 (≈ 34 × an uncached F1 run when nothing is cached, and 80
requests per note), D2 ≈ 2.1 ×, D6 ≈ 1.6 ×, D7 ≈ 1.1 ×, D3 ≈ 0.9 ×, D2p ≈ 0.8 ×; F3 and H5 sit at ≈ 2.8 × because they re-emit the
note. Output tokens, not input, decide the ranking among the one-call designs, which is why the verdict format (verbose
vs positives-only vs entity-grouped) is itself an experimental variable. The ratios are stable from 1k- to 3k-token notes
because every term scales with N or K; the absolute numbers scale linearly.

### 3.7 Summary matrix

| ID | Family | Targets (failure rows of §1.3) | Human labels needed | LLM calls / note | Priority |
|---|---|---|---|---|---|
| H1 Union | fusion | all misses | none | 1 | P1 |
| H2 Label-routed | fusion | rare label, unsupported label | none (freq table) | 1 (restricted output) | P1 |
| H3 Confidence-gated | fusion | union FPs | small dev | 1 + short verifications | P1 |
| H4 Stacking | fusion | union FPs | 100–300 | 1 | P2 |
| H5 TC-first editor | cascade | LLM omissions, grounding, TC tail, TC FPs | none | 1 (long output) | P1 |
| H6 LLM-first + net | cascade | — | none | 1 | P3 |
| H7 Selective invocation | cascade | cost | none | fraction | P1 |
| H8 Audit loop | cascade | residual misses | none | 1–2 | P2 |
| H9 Silver distillation | training | no-data regime | none | 1 per silver note (one-off) | P1 |
| H10 Rare-label injection | training | rare label, formats | base set | generation only | P1 |
| H11 Disagreement AL | training | label efficiency | budgeted | 1 per pool note | P2 |
| H12 Co-refinement loop | training | residual errors of both | 100–200 (dev) | per round | P2 |
| H13 OPF head extension | training | fixed taxonomy | 25–400 | 0 | P1 |
| H14 Hint-conditioned TC | representation | trust calibration | silver | 1 | P3 |
| H15 Format ablation | grounding | LLM grounding | none | 1 | P1 |
| H16 Zero-shot span model | third component | unsupported labels, gating | none | 0 | P3 |
| D0 Candidate coverage | decision | (analysis: recall ceiling of D) | none | 0 | P1 |
| D1 Per-candidate verdicts | decision | omission, grounding | none | K (≈ 80, short) | P2 |
| D2 Batched verdicts (+D2p/D2b) | decision | omission, grounding, H5 output cost | none | 1 (bounded output) | P1 |
| D3 Segment gate → D2 | decision cascade | cost, routing without a TC | none (D3c: silver) | 1 cheap + 1 partial | P2 |
| D4 Logprob scoring | decision | calibration, recall-first thresholds | small dev | K (local: GPU-s) | P2 |
| D5 Distilled decision classifier | decision / training | inference cost, no-data regime | none (silver verdicts) | 0 (one-off D2 pass) | P3 |
| D6 Entity-level verdicts | decision | repeated-mention omissions, output cost | none | 1 | P1 |
| D7 Over-generate → decide | decision cascade | K, precision over a noisy proposer | none | 1 cheap + 1 | P2 |
| S1–S3 Surrogation | surrogation | leakage / utility | none | 0–1 | P2 |

---

## 4. Research questions, hypotheses, experiments

| RQ | Hypothesis (prediction) | Experiment | Primary metrics / analysis |
|---|---|---|---|
| **RQ1** Are the error sets complementary? | J(FN_LLM, FN_TC) is low; conditional miss rates diverge from marginals; R_∪ − max(R_LLM, R_TC) ≥ 3 points, concentrated in buckets 0 / 1–10 (TC misses) and in repeated mentions / long notes (LLM misses) | **E1**: run S_LLM (best format from E8) and S_TC (full training) on the test split; compute §2 statistics per label, bucket, note type, mention form; manual error taxonomy on a stratified sample (150 FN per system, 200 FP) coded by two annotators (κ reported): rare label · format · context-dependent · boundary · grounding · label confusion · over-redaction · gold gap | Venn/Jaccard of FN sets; heatmap label × system miss rate; conditional-miss bar chart; taxonomy distribution |
| **RQ2** Which fusion recovers the headroom at what precision cost? | H3/H5 reach recall within 1 pt of R_∪ with precision within 2 pts of the better single; H1 loses more precision; H8 adds little | **E2**: H1, H2 (τ sweep), H3 (θ sweep), H5 (+H5b), H7 (ρ sweep), H8; thresholds tuned on dev, reported on test; 3 LLM runs × 3 TC seeds | Pareto (recall vs precision; recall vs cost); per-label tables; paired bootstrap Δrecall/ΔF1 vs best single (1,000 resamples, Holm); McNemar on span-level miss indicators |
| **RQ3** Where does the hybrid help most as a function of human-label budget? | Hybrid gain is largest at N ≤ 100 and shrinks but does not vanish at full supervision (tail labels persist) | **E3**: TC trained on N ∈ {0, 25, 50, 100, 200, 400, all} notes × 3 seeds; S_LLM constant; hybrids H2/H3/H5 at each N; H9 silver TC using the remaining pool | learning curves (overall, per bucket, rare labels); crossover point; rare-label recall vs n_ℓ |
| **RQ4** Can LLM-manufactured supervision close the gap to full supervision? | Silver + rare-label injection ≥ 90% of full-supervision F1 on the tail; disagreement AL beats uncertainty and random on tail labels | **E4**: H9 variants (single / vote-2 / vote-1 / iterated); H10 sweeps (0/10/50/200 per label; template vs LLM insertion; counterfactual names); H11 AL simulation (5 rounds × 20 notes); H13 head-extension options (a)–(d) | F1 vs full supervision; rare-label recall vs injected count; AL curves and AUC; FP on untouched split |
| **RQ5** What is the recall–cost frontier? | H7 attains ≥ 80% of the union gain at ≤ 30% of the LLM cost | **E5**: per-note cost logging for every configuration; H7 ρ sweep; segment-level routing | $/1k notes, tokens/note, latency; recall of unflagged notes |
| **RQ6** Do results transfer across sources, note types and institutions? | Hybrid gains persist (or grow) under shift; TC alone degrades most | **E6**: train on MTSamples-injected → test on Asclepius-injected (and reverse); held-out note types; unseen institution lexicon; transfer to i2b2 2014 when access reopens | Δ across sources per system; institutional-label recall |
| **RQ7** Does hybrid surrogation reduce leakage while preserving utility? | S3 ≥ S1 on realism and consistency with equal leakage; detector re-run equivalent within ±2 pts | **E7**: S1–S3 on hybrid-detected spans; attacker test; downstream tasks | leakage; attacker precision; consistency; TOST; utility drift |
| **RQ8** How much of the LLM miss rate is format-induced? | F3/F4 cut alignment failures to near zero and reduce total misses by a third; semantic omissions remain | **E8**: H15 formats F1–F4 (F5 optional), same model, same guideline, 3 runs | parse-failure rate; alignment-failure rate; recall decomposition; output tokens |
| **RQ9** Does decision-style prompting over explicit candidates remove the LLM's omission and grounding failures, and at what cost? | Union proposer relaxed R_C ≥ 0.98; D2 recall ≥ F3 recall with zero alignment failures at ≤ 2.1 × F1 cost (D2p ≤ 0.8 ×); D6 = D2 in recall at lower cost; D1 ≈ D2 in decision quality at ≥ 4 × cost and 80 × the requests; D4 scores calibrate (ECE < 0.05 after scaling) and trace a recall–precision curve that dominates fixed verdicts; D7 ≈ D2 only where the cheap proposer's R_C ≥ 0.97 | **E9**: D0 for proposers (a)–(e); D1 on a 200-note dev subset; D2 / D2p / D2b / D6 / D7 on dev, 3 runs; D3 (fast gate) and D3c threshold sweeps; D4 on `llm-local`; same guideline and strong model as E8; costs from the per-call log and Appendix D | R_C per label / bucket; decision P/R over candidates; end-to-end recall decomposition (candidate miss / decision miss / boundary); calibration (ECE, risk–coverage); $/1k notes, calls/note, hours at concurrency 1 |

Pre-specification: thresholds and prompts are frozen on dev before test evaluation; every configuration is a versioned YAML
under `configs/experiments/`; prompt text is hashed and logged with each run.

---

## 5. Evaluation protocol

1. **Matching**: strict, relaxed, and type-agnostic relaxed (§2), implemented once in `note_deid.eval` and unit-tested
   against hand-built cases; i2b2-2014 semantics for the gated benchmark.
2. **Metrics**: entity-level P/R/F1 micro and macro, per subtype, per category, HIPAA-subset; token-level P/R/F1; document-level
   leak rate (fraction of notes with ≥ 1 residual gold span, strict and relaxed); **recall is primary**, precision and the
   information-loss proxy (fraction of non-PHI tokens redacted; clinical-term redaction count using a drug/condition lexicon)
   are secondary.
3. **Frequency analysis**: every per-label number is also aggregated by bucket B of n_ℓ in the *TC training split*; the
   benchmark's frequency profiles make bucket membership controllable (§6).
4. **Statistics**: paired bootstrap over documents (1,000 resamples) for differences; Holm correction across configurations;
   3 LLM runs (T = 0, different days/seeds where possible) and 3 TC seeds; mean ± sd; TOST for equivalence claims.
5. **Gold adjudication**: LLM-only spans that survive review are candidates for gold gaps (Palacios: 414 → confirmed PHI).
   Protocol: a human reviews a stratified sample of LLM-only spans; confirmed gaps are added to a *v1.1 gold*; results are
   reported on both gold versions.
6. **Cost**: per call — model, prompt hash, input/output tokens, cache-read tokens, USD (LiteLLM `completion_cost`),
   latency; per configuration — $/1k notes, calls per note, and wall-clock at concurrency 1 (requests, not tokens,
   bound the per-candidate designs of §3.6). Closed-form planning estimates come from `note_deid.llm.costmodel`
   (Appendix D) and are replaced by logged counts once a configuration has run.
7. **Reporting checklist**: model identifiers and dates, prompt hash, temperature, seeds, benchmark version, split hashes,
   thresholds, hardware. (The EMNLP 2025 survey's main complaint is inconsistent reporting.)

---

## 6. Benchmark and data plan

### 6.1 Tiers

- **Tier B — primary: PHI-injection benchmark (working name `synphi`)** built from open, PHI-free notes (MTSamples;
  Asclepius synthetic notes; optionally Synthetic4Health after license check). Justification: detector behaviour on
  surrogate-substituted text is statistically equivalent to original text (arXiv 2608.03172); label frequencies become
  controllable; the corpus is shareable and reproducible; the gated standard is currently unreachable.
- **Tier A — gated, deferred**: i2b2/n2c2 2014 (primary external validation once the DBMI portal reopens), i2b2 2006,
  CEGS N-GRID 2016, PhysioNet `deid` nursing notes, MIMIC-IV-Note for re-injection. Loaders and label maps are prepared so
  E1–E6 can be re-run unchanged.
- **Tier C — auxiliary**: ASQ-PHI (over-redaction on PHI-free queries), pii-shield / Nemotron-PII (non-clinical
  pre-training), SHIELD if released, RedactionBench (contextual subset), MEDDOCAN (Spanish generalization, optional).

### 6.2 `synphi` specification (v0)

1. **Source cleaning**: strip MTSamples placeholders ("Dr. X", "(Patient name)", "[DATE]", template DOBs); run OPF, regex
   and one LLM pass to flag residual PHI-like strings; drop notes that cannot be cleaned; keep note type and specialty metadata.
2. **Entity generators** per subtype: Faker (names incl. international and uncommon; street/city/state/zip/country; phones,
   fax, emails, URLs, IPs; SSN-shaped, license, VIN); Synthea for coherent patient bundles (name, DOB, address, MRN,
   providers, organizations); curated lists for hospitals/organizations/professions; institution lexicon (department codes,
   building names, local abbreviations) for the INSTITUTIONAL group; ages incl. ≥ 90.
3. **Insertion modes**: (a) slot templates — patient identification blocks, letterheads, signature/"Dictated by"/"cc:"
   lines, appointment lines; (b) in-text replacement of generic mentions ("the patient" → name at a set rate, "the
   hospital" → hospital name); (c) LLM contextual insertion — rewrite a chosen sentence to mention specified entities
   naturally, returned as tagged text and validated (entities verbatim, no other change); (d) document-level consistency —
   repeated mentions with variants (first name only, last name only, initials, possessive, one misspelling), coherent dates
   (admission < discharge; DOB consistent with age).
4. **Frequency profiles** (config): `i2b2like` (subtype proportions approximating the i2b2 2014 training distribution:
   DATE ≫ DOCTOR > HOSPITAL > PATIENT > AGE > MEDICALRECORD > CITY/STATE/PHONE/STREET/ZIP/PROFESSION/IDNUM > ORGANIZATION >
   USERNAME > COUNTRY, with FAX, EMAIL, URL, DEVICE, BIOID, HEALTHPLAN, IPADDR, SSN, VEHICLE, LICENSE, ACCOUNT, LOCATION-OTHER at
   ≤ 10 instances — exact counts to be copied from the corpus paper when the DUA copy is available); `uniform`; `tailstress`
   (tail labels up-weighted, evaluation only); `decoupled` (train profile ≠ test profile, to test rare-label generalization).
5. **Gold construction**: offsets by construction; assertion that every injected string is labeled at every occurrence;
   pre-existing residual PHI-like strings resolved before injection (step 1).
6. **Validation of realism**: human spot-check of 100 notes (naturalness of insertion, 5-point scale); a discriminator test
   (can a classifier tell LLM-inserted sentences from original ones?); detector-equivalence check against a real corpus when
   Tier A becomes available; results on real data are the ultimate check and are planned in RQ6.
7. **Splits & versioning**: split by source document (no leakage), stratified by note type; fixed seeds; `synphi-v0.1`
   with a manifest (source ids, generator config, seed, profile) so the corpus can be regenerated exactly; distribution as
   generator + manifest (and as text where source licenses permit; Asclepius is CC-BY-NC-SA).
8. **Stated limitations**: distribution shift from real notes; injected PHI is not "organically" embedded; no real
   institution-specific idiosyncrasies (simulated via the lexicon).

### 6.3 Label mapping

`labels.py` holds the canonical 28 leaf labels, the 7 categories, the HIPAA subset, and mapping tables to OPF-8 and
OpenMed-54 (normalized at load time against the model's `id2label`). Unmapped subtypes (AGE, PROFESSION, HOSPITAL,
ORGANIZATION, USERNAME, IPADDR, COUNTRY for OPF) are exactly the "unsupported" set used by H2/H3/H13.

---

## 7. Backends and infrastructure

### 7.1 Unified interfaces (`src/note_deid`)

```
Doc(doc_id, text, spans: list[Span], meta)          Span(start, end, label, text, source, score)
Detector.detect(docs) -> list[list[Span]]           # LLMDetector, TCDetector, ZeroShotDetector
Fusion.fuse(doc, outputs: dict[name, list[Span]]) -> list[Span]   # H1–H5, H7 policies
Evaluator.score(gold_docs, pred_docs) -> Report     # metrics of §5, per label / bucket, bootstrap
CostLog.record(call)                                # tokens, USD, latency, prompt hash
```
Everything is configuration-driven (`configs/experiments/*.yaml`): detector configs, fusion policy, benchmark version,
splits, seeds. Runs write `results/<exp>/<run-id>/{predictions.jsonl, metrics.json, cost.json, config.yaml}`.

### 7.2 LLM backend (LiteLLM)

- `litellm.completion()` with model aliases from `configs/litellm.models.yaml` (default aliases: `llm-strong`, `llm-fast`;
  optional `llm-local` pointing at an OpenAI-compatible vLLM/Ollama server). Keys from environment only.
- JSON-schema `response_format` where the provider supports it; otherwise instructed JSON + tolerant parsing.
- Disk cache keyed by (model, params, prompt hash, text hash) — reruns are free; cost tracking via `completion_cost`.
- Retries with backoff; **request concurrency defaults to 1** (rate-limit friendly), configurable per run.
- Prompt templates under `prompts/` (guideline versions, format variants F1–F4, editor prompt, verifier prompt, and the
  family-D decision prompts: per-candidate verdict, batched verdicts over markers, segment gate).
- **Prompt caching**: the guideline + instructions are the stable system prefix (hash-stable rendering, no per-call
  text before it) so that every provider that caches prefixes serves it at the cache-read price; `LLMCall` is to gain a
  `cache_read_tokens` field, because the per-candidate designs (D1/D4) are only affordable when the prefix is cached
  and the cost log must show that it was. Cache reads also do not count toward input-token rate limits on the
  Anthropic API, which matters at concurrency 1.
- **Log-probabilities** (D4) are only available on OpenAI-compatible endpoints and the local vLLM server (`llm-local`);
  the client exposes them as an optional `logprobs` request parameter and leaves the score `None` elsewhere.
- `note_deid.llm.costmodel`: closed-form per-note token/cost estimates for F1/F3/H5 and D1–D7 (Appendix D).

### 7.3 Token-classification backend (HF Transformers + OPF)

- `AutoModelForTokenClassification` for OPF (`openai/privacy-filter`; native 128k context, no chunking), DeBERTa-v3
  (512-token windows with stride), OpenMed-PII; BIOES/BIO decoding with span probabilities retained.
- Training paths: (i) `opf train train.opf.jsonl --output-dir …` with JSONL produced by `synphi export-opf`
  (`schema.to_opf_record`; OPF label space, mapped labels only); (ii) HF `Trainer` for head extension (H13) and for
  DeBERTa baselines; seqeval for token-level checks, our evaluator for entity-level numbers. Convention: path (ii)
  (`configs/train/opf_finetune.yaml`) trains the 28-label head on the unified JSONL, so its checkpoints emit canonical
  subtypes; a native-OPF checkpoint from path (i) is evaluated through `label_map: opf_category` at category level
  only, and the runner refuses a checkpoint whose labels are neither canonical nor mapped.
- Model registry: `configs/train/*.yaml` (base checkpoint, label set, lr, epochs, batch, max length, precision, seeds).

### 7.4 Hardware matrix

| Workload | Apple Silicon (M-series, PyTorch MPS) | 4×A100 80GB |
|---|---|---|
| OPF inference | fine (1.5B, 50M active) | fine |
| OPF fine-tuning | small runs only; MoE kernels on MPS unverified → CPU fallback for smoke tests | full fine-tunes, bf16, `torchrun --nproc_per_node=4` / accelerate DDP |
| DeBERTa-v3-base | fine-tune feasible (small batch) | fine |
| DeBERTa-v3-large / OpenMed-434M | inference; fine-tuning slow | fine |
| Local LLM (optional) | Ollama / MLX ≤ 8B | vLLM 32–70B |
| Sweeps (E3/E4) | no | yes |

### 7.5 Reproducibility

Pinned dependency lock (`uv.lock` or `requirements.lock`), seeds, dataset manifests, prompt hashes, model identifiers,
and a `results/` layout that stores predictions so every metric can be recomputed offline.

---

## 8. Phased roadmap

| Phase | Deliverables | Depends on |
|---|---|---|
| **P0 (this session)** | `literature.md`, `framework-designs.md`, `docs/datasets.md`, skeleton, schema/labels | — |
| **P1 — foundation** | `synphi-v0.1` (≈ 2k notes, `i2b2like` + `decoupled` profiles); evaluator + tests; port of S_LLM (from prior code) with formats F1–F4; S_TC training on OPF and DeBERTa; **E1 + E8** → complementarity report (the paper's Figure 1) | prior artifacts imported; API keys; A100 or Mac for TC |
| **P2 — inference-time hybrids** | H1, H2, H3, H5, H7, H8 → **E2, E5**; D0, D2 (+D2p/D2b), D6, D7, then D1/D3/D4 → **E9** | P1 |
| **P3 — training-time hybrids** | H9, H10, H13 (+ H11, H12, D5 if budget) → **E3, E4** | P1, A100 sweeps |
| **P4 — surrogation** | S1–S3 → **E7** | P2 |
| **P5 — robustness & paper** | E6 (cross-source, note type, institutional; i2b2 transfer if access), gold adjudication, ablations, writing | P2–P4 |

Compute/cost envelope (to be measured in P1): ≈ 2k test notes × ~12 LLM configurations × 3 runs ≈ 70k calls; with caching
and dev-subset tuning the paid volume should stay well below that; all LLM runs are cached and resumable.

---

## 9. Publication plan

- **Working title**: *Complementary failure modes: hybrid LLM + token-classifier de-identification of clinical notes under
  label scarcity.*
- **Contributions**: (1) first quantitative error-complementarity analysis between LLM and trained token classifiers for
  de-identification, per label and per training frequency; (2) a taxonomy and evaluation of hybrid designs (fusion,
  cascade, training-time coupling, grounding, and decision-style prompting over explicit candidates); (3) results across human-label budgets, including the zero-label regime;
  (4) a reproducible PHI-injection benchmark with controllable label frequencies; (5) cost–recall frontiers for deployment;
  (6) surrogation with hybrid detection (if P4 completes).
- **Positioning**: vs Palacios 2026 (homogeneous LLM ensembles fail; ours are heterogeneous and routed), SHIELD (distillation
  baseline; we address its rare-category gap), locally augmented ensembles (no LLM), JSL LLM-in-the-loop (closed data).
- **Venues** ▶ (recommended: JAMIA or JBI for the full study; ML4H / CHIL / clinical-NLP workshop for an early version).
- **Figures**: F1 FN-set overlap + label × system miss heatmap; F2 recall–precision Pareto of hybrids; F3 learning curves;
  F4 rare-label recall vs training count; F5 recall–cost frontier; F6 extraction vs decision designs: recall decomposition
  (candidate / decision / boundary / alignment misses) against $/1k notes; tables per subtype.
- **Threats to validity**: synthetic-injection realism; LLM version drift (mitigated by pinned model ids, dates, cached
  outputs); gold gaps; English only; single taxonomy.

---

## 10. Decision log

**Decided (2026-09-14)**: canonical taxonomy = i2b2-2014 subtypes; primary benchmark = `synphi` on MTSamples + Asclepius;
commercial LLM APIs allowed for all in-scope data; TC base = OPF with DeBERTa-v3 as control; recall-first protocol; no data
files in git; prior artifacts imported after the skeleton (Appendix A); serialized, single-agent execution.

**Decided (2026-09-14, license)**: code under Apache-2.0 (`LICENSE`, `NOTICE`; matches OPF, Transformers and OpenMed);
documents, prompts, guidelines and fully synthetic data under CC BY 4.0 (`LICENSE-DOCS`); third-party corpora keep their
own terms, so `synphi` is released as recipe + manifest with built text only for the synthetic subset, guideline examples
must be synthetic, and released checkpoints are trained only on data whose terms allow it (`docs/datasets.md` §4).
`CITATION.cff` added for the software citation.

**Open** ▶ (with recommended defaults):
1. INSTITUTIONAL label group in `synphi` — recommended **yes** (flag).
2. LLM aliases: `llm-strong` and `llm-fast` model choices — recommended one frontier model and one cheap model from
   different providers, pinned by dated identifiers in `configs/litellm.models.yaml`.
3. Human annotation budget for H11/H12 — recommended ≥ 100 notes adjudicated (or none: simulate with gold).
4. Which prior artifacts exist and in what format (guideline version, annotation format, TC checkpoint) — Appendix A.
5. Target venue and timeline.
6. Candidate proposer for the decision-model family (§3.6) — recommended **union of surface-form regex/capitalization
   rules and TC spans** (proposer (e) without the LLM member), with D0 coverage reported per label before any D run.
7. Verdict output format for D2 — recommended **verbose (one verdict per candidate)** as the primary condition, with
   positives-only (D2p) and entity-grouped (D6) as the cost arms; positives-only becomes the default only if E9 shows no
   recall gap.
8. Where D4 (log-probability scoring) runs — recommended **local vLLM only** (`llm-local`), since the commercial aliases
   in `configs/litellm.models.yaml` either do not expose log-probabilities or make per-candidate calls cost ≈ 10 × F1.

---

## Appendix A — Integrating the prior experiments

| Artifact | Destination | Needed from the user | Used by |
|---|---|---|---|
| Annotation guideline (current version) | `prompts/guideline_v0.md` (+ the DSPy program / GEPA run config that produced it) | text, DSPy program, GEPA config | S_LLM baseline, H12 G₀, H9 labeler |
| LLM prompt(s) and output parser | `prompts/`, `src/note_deid/llm/` | prompt text, model id, parsing rules | H15 format F1 |
| Annotated notes | `data/annotations/` (gitignored) + converter to `Doc` JSONL | format (brat / XML / JSON), label set | dev₁/dev₂, adjudication, E3 learning curves if enough |
| TC training config / checkpoint | `configs/train/`, `models/` (gitignored) | base model, hyperparameters | S_TC baseline |
| Prior results tables | `experiments/prior/` | CSV/markdown | baseline numbers, error examples for §1.3 |

## Appendix B — LLM prompt/output variants (sketch)

- **F1 (current)**: system = guideline; user = note; output `{"spans": [{"text": "...", "label": "..."}]}`; alignment =
  exact search over all occurrences.
- **F2**: same, plus `"before": "<≤20 chars of left context>"` per span; anchored search; fuzzy fallback.
- **F3**: output = the full note with `<PHI type="LABEL">…</PHI>` tags; validated by stripping tags and comparing to the
  input; any other change → reject.
- **F4**: input split into numbered segments; output = tagged segments by number (bounded output, parallelizable).
- **Editor (H5)**: input = note pre-tagged by the TC; instructions = add / remove / retype; output as F3 or as an edit list
  `{"add": [...], "remove": [...], "retype": [...]}` keyed by occurrence index.
- **Verifier (H3)**: input = candidate span with ±200 characters, label definition; output `{"is_phi": bool, "label": ...}`.
- **Decision, per candidate (D1/D4)**: the verifier prompt applied to every proposed candidate; D4 asks for a single answer
  token (`yes`/`no`, or a label letter) and reads its log-probability as the score.
- **Decision, batched (D2)**: input = the note with numbered markers around every candidate (`⟦12|03/04/2019⟧`); output
  `{"decisions": [{"i": 12, "phi": true, "label": "DATE", "extend": [0, 0]}, ...]}`; **D2p** = only accepted
  candidates as `12:DATE` lines; **D2b** = D2 plus `"additions": [{"text": ..., "label": ...}]` for PHI without a
  marker (aligned like F1); **D6** = one index per distinct string with `"except": [occurrence indices]`.
- **Segment gate (D3)**: input = numbered segments (`[1] ...`); short system prompt (label list, one rule); output = the
  indices of segments that contain PHI, optionally with the labels suspected.

## Appendix C — H3 arbitration (pseudo-code)

```
for cand in merge_extents(Y_llm ∪ Y_tc):
    if cand in Y_llm and cand in Y_tc:            accept(cand, label=resolve(cand))
    elif cand in Y_tc:                            # TC-only
        if cand.tc_score >= θ_hi:                 accept(cand)
        elif verify_llm(cand):                    accept(cand)
        else:                                     drop(cand)
    else:                                         # LLM-only
        if cand.label in UNSUPPORTED or n[cand.label] < τ:   accept(cand)
        elif tc_forced_prob(cand) >= θ_lo or verify_llm(cand): accept(cand)
        else:                                     drop(cand)
```


## Appendix D — Per-note token and cost model for the LLM designs

Produced by `python -m note_deid.llm.costmodel` (no API calls). Assumptions are explicit parameters so the table can be
regenerated with logged counts once a configuration has run; the defaults below are planning estimates for a ~1k-token
note. Prices are Anthropic list prices read on 2026-11-02: the "strong" role is priced at the Sonnet-5 point (2 / 10 USD
per 1M input / output tokens — the model behind the `llm-fast` alias in `configs/litellm.models.yaml`; the OpenAI model
behind `llm-strong` was not re-priced in this session and is supplied with `--input-per-m/--output-per-m`), the "fast"
role at the Haiku-5.5 point (0.10 / 0.50); cache reads at 0.1 × the input price; batch API 0.5 ×. Like the model
identifiers themselves these are placeholders to re-verify before any number enters the paper. Hours are wall-clock at
concurrency 1 with 1 s request overhead and 60 output tokens/s.

| Symbol | Default | Meaning |
|---|---:|---|
| N | 1,000 | note tokens |
| G | 1,500 | system prompt: guideline + format instructions (the cacheable prefix) |
| P | 20 | gold PHI mentions per note |
| K | 80 | candidates from the surface-form proposer; K' = 30 from the cheap over-generating proposer (D7) |
| S, φ | 25, 0.4 | gate segments per note and the fraction flagged (D3) |
| rep | 1.6 | occurrences per distinct candidate string (D6) |
| per-candidate tokens | 130 / 6 / 10 / 5 | snippet + framing (D1, D4) / inline marker (D2, D6) / verbose verdict / positives-only verdict |
| per-span tokens | 15 / 12 | F1 JSON item / inline tag pair (F3, H5) |

**Default workload** (USD per 1,000 notes; "cache" = guideline served from the prompt cache; "x F1" on the cached column):

| ID | Design | calls/note | input (uncached) | input (cache read) | output | USD/1,000 no cache | USD/1,000 cache | xF1 | USD/1,000 cache+batch | h/1,000 @1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | extraction, JSON strings (reference) | 1.0 | 1,000 | 1,500 | 300 | 8.00 | 5.30 | 1.00 | 2.65 | 1.7 |
| F3 | extraction, inline tags | 1.0 | 1,000 | 1,500 | 1,240 | 17.40 | 14.70 | 2.77 | 7.35 | 6.0 |
| H5 | TC-first editor, full tagged text out | 1.0 | 1,240 | 1,500 | 1,240 | 17.88 | 15.18 | 2.86 | 7.59 | 6.0 |
| D1 | one decision call per candidate | 80 | 10,400 | 120,000 | 800 | 268.80 | 52.80 | 9.96 | 26.40 | 25.9 |
| D2 | batched decisions over marked candidates | 1.0 | 1,480 | 1,500 | 800 | 13.96 | 11.26 | 2.12 | 5.63 | 4.0 |
| D2p | D2, positives-only output | 1.0 | 1,480 | 1,500 | 100 | 6.96 | 4.26 | 0.80 | 2.13 | 0.7 |
| D2b | D2 + free-text additions for unproposed PHI | 1.0 | 1,480 | 1,500 | 830 | 14.26 | 11.56 | 2.18 | 5.78 | 4.1 |
| D3 | fast-model segment gate, then D2 on flagged | 2.0 | 1,692 | 1,900 | 350 | 7.55 | 4.81 | 0.91 | 2.41 | 2.2 |
| D3c | classifier segment gate, then D2 on flagged | 1.0 | 592 | 1,500 | 320 | 7.38 | 4.68 | 0.88 | 2.34 | 1.8 |
| D4 | per-candidate logprob scoring (API variant) | 80 | 10,400 | 120,000 | 80 | 261.60 | 45.60 | 8.60 | 22.80 | 22.6 |
| D5 | distilled decision classifier (inference) | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 |
| D6 | D2 with one decision per distinct string | 1.0 | 1,480 | 1,500 | 500 | 10.96 | 8.26 | 1.56 | 4.13 | 2.6 |
| D7 | fast over-generation, then strong D2 | 2.0 | 2,180 | 3,000 | 750 | 8.83 | 6.00 | 1.13 | 3.00 | 4.0 |

**Long notes** (N = 3,000, P = 60, K = 240, K' = 90, S = 75; same prices): the ranking is unchanged because every term
scales with N or K — F1 15.30, F3 43.50, H5 44.94, D1 158.40, D2 33.18, D2p 12.18, D2b 34.08, D3 13.83, D3c 13.45,
D4 136.80, D6 24.18, D7 17.37 USD per 1,000 notes with caching. With the `llm-strong-alt`-class price point (5 / 25 USD)
every USD figure is 2.5 × higher and the ratios are identical.

**Reading the table.**

- Among the one-call designs the *output* term decides the order: D2's 800 verdict tokens cost more than F1's 300
  span-item tokens; D2p (positives only) and D6 (one verdict per string) recover most of that; F3 and H5 pay ≈ N output
  tokens to re-emit the note. The verdict format is therefore an experimental variable in E9, not an implementation detail.
- The per-candidate designs (D1, D4) are dominated by *requests*, not tokens: 80 calls per note at concurrency 1 is a day
  per 1k notes, and without prompt caching the guideline is re-read 80 times (≈ 34 × an uncached F1). They are reference
  conditions
  and the H3 verifier, not primary detectors.
- Gating (D3/D3c) and over-generation (D7) bring the strong-model cost to ≈ F1 while keeping D2's grounding properties;
  their risk is the recall of what the gate or proposer does not pass on, which E9 reports separately.
- D5 moves the whole family to a local encoder at inference; its cost is one D2 pass over the silver pool (for a 5k-note
  pool at the default workload ≈ 56 USD with caching).
- Batch processing halves every figure and suits the offline evaluation runs; caching needs calls that share the prefix
  within the cache lifetime (5 min by default), which the serialized runner guarantees.
