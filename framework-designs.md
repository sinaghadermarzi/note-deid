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
the task shrinks to a verdict, offsets are known by construction for every marker-derived span, and output length is
bounded by the number of candidates rather than by the note. The H3 verifier (Appendix B) is already a decision model,
applied only to disputed candidates; the H5 editor is a decision model in disguise (a verdict per pre-tagged span plus
additions) that pays for its verdicts by re-emitting the whole note. Family D makes the decision model the primary LLM
role. In the project's earlier notes this is "model 1 (decision)", as opposed to the extraction model.

**Why it may fix what extraction cannot — and what it trades for it.** The LLM failures of §1.3 split into (i)
*omission* — the model never emits a span that is in the text — and (ii) *grounding* — it emits a non-verbatim string or
an ambiguous repeat. Decision prompting removes (ii) for every marker-derived span (alignment failures can re-enter only
through the free-text additions of D2b and through boundary edits, which are logged as separate terms below) and turns
(i) from recall-from-memory (enumerate everything you noticed) into recognition (answer for each marked item) — the
argument behind H5, now with a bounded output. The price is a **candidate miss**: a gold mention that no proposer offers
can never be recovered. Family D is therefore itself a hybrid whose fusion point is the *candidate set*: it trades the
LLM's rows of §1.3 (omission, grounding, correlated over-redaction) for the proposer's rows (format brittleness,
labels without a lexicon, institution-specific strings); T1's complementarity argument applies to it in reverse, and RQ9
asks whether the trade is favourable. Every design is judged on three numbers — candidate coverage R_C, decision quality
over candidates, and cost per note — and end-to-end misses are decomposed, extending RQ8.

*Definitions (implemented once in `note_deid.eval`).* A candidate matches a gold span **strictly** if (s, e) are identical
and **relaxedly** if they overlap; R_C is label-agnostic (the proposer carries no label) and is reported strict / relaxed,
per label, per frequency bucket, per mention form (§2) and — on `synphi` — per insertion mode (§6.2). *Candidate gold*: a
candidate is positive iff it relaxed-matches a gold span; its target label is that span's label (longest overlap on ties).
*Decomposition per gold span*, in order: candidate miss (no overlapping candidate) → decision miss (overlapping candidate
rejected) → label error (accepted, wrong label) → boundary (accepted, right label, extent ≠ gold after extend/split edits)
→ alignment (D2b additions only). *Boundary-recoverable* = relaxed hits that become strict hits under the extend/split
edits allowed by the design; it is the strict ceiling of D2. *Positives-only outputs* (D2p) end with a terminator
`END <n>` (n = number of positives); a missing or mismatched terminator is a parse failure, so truncation is never read as
rejection, and the D2 vs D2p gap is a candidate-level McNemar test paired by candidate id across runs.

**Design axes.**

- *Unit of decision*: candidate span · distinct string (entity) · segment/sentence · numbered token (D8, the
  proposer-free control).
- *Candidate proposer*: (a) **surface-form** — regex shapes (dates, phones, fax, emails, URLs, IPs, SSN/ID shapes, ZIPs,
  ages), capitalized token runs not at sentence start, titled names (Dr./Mr./Ms.), digit runs, values on header lines
  (`HEADER_LINE` in `hybrid/routing.py`, with the value — not the whole ALL-CAPS line — as the run); recall-maximizing,
  ≈ 60–100 candidates per 1k-token note; (b) **TC-based** — TC spans plus uncovered regex cues (`RARE_CUE_PATTERNS`) plus
  low-confidence tokens, ≈ 30; (c) **zero-shot span model** (GLiNER, H16); (d) **cheap LLM over-generation** (an
  extraction pass told to list anything that could be PHI); (e) the union of (a)–(d). Coverage is measured per proposer
  before any decision design runs (D0).
- *Decision function*: strong LLM with verdict lines · LLM with a single-token answer scored by log-probability
  (calibrated) · distilled encoder over the marked context · local model with prefix caching.
- *Context*: ±200-character snippet vs the whole note (document-level cues: a bare first name is PHI because the full name
  is in the header; letterheads; signature blocks). D1 vs D1n isolates this axis at a fixed decision function.
- *Rendering*: inline markers in the text vs an out-of-band candidate table after an unmodified note (D2 vs D2t): markers
  change the text the model reads and are themselves a "this is suspicious" prior.
- *Batching*: one call per candidate · one call per note · one call per flagged segment.

**D0 — Candidate-coverage analysis (no LLM).** For each proposer (a)–(e): R_C strict/relaxed per label, bucket and
mention form against gold on dev; the boundary-recoverable share; the candidate-to-gold ratio (the noise a decision model
must absorb); and the candidate repetition factor rep (occurrences per distinct string) that D6's saving depends on. Fixes
the recall ceiling of every D design and the K that drives their cost. **Provenance caveat**: the surface-form rules and
the `synphi` injection engine share regex and template provenance — `HEADER_LINE` names exactly the keys after which the
header and signature templates place entities, and the regex cues mirror the Faker generators — so R_C on template
insertions is near 1 by construction and says little about real notes. R_C is therefore reported separately by insertion
mode — (a) slot templates, (b) generic-mention replacement, (c) LLM contextual insertion — and on the pre-existing
residual PHI-like strings flagged in §6.2 step 1 (real text); the headline figure is the mode-(c) + residual number, and a
`template-heldout` profile (§6.2) is written after the proposer rules are frozen (hash of `candidates.py` recorded).
Requires: a proposer module (`llm/candidates.py`, planned) sharing its regexes with H7; generator forms that stress the
proposer — lowercase / dictation-style names, possessives, one misspelling, non-`en_US` Faker locales (§6.2(d), currently
unimplemented); and a case-folded ablation (lowercase the note before proposing; the R_C drop per label is the
capitalization-dependence probe). Expected: relaxed R_C ≥ 0.98 on template insertions (by construction) and ≥ 0.95 on
contextual insertions for the union proposer at K ≈ 80–120; strict R_C lower on multi-token names and suffixed dates
unless candidates are maximal runs; gaps concentrate in lowercase names, PROFESSION, LOCATION-OTHER, HOSPITAL
abbreviations, partial and relative dates ("May 2019", "last spring") and department/building codes (Palacios 2026).
**P1** (cheap; prerequisite for the family)

**D1 — One verdict call per candidate (the H3 verifier generalized).** For every candidate: system = guideline + verdict
instruction (a cacheable prefix), user = snippet with the candidate bracketed and an optional proposed label (`null` for
shape-less candidates) → `{"is_phi": bool, "label": ...}`. Variant **D1n**: the whole note sits in the cached prefix
(system = guideline + note, user = one candidate), the API counterpart of the local D4 variant; E9 runs D1 and D1n on the
same 200-note subset so the D1→D2 difference can be attributed to context versus batching. Targets: grounding; the
reference decision function (a snippet verdict does not itself address omission — the proposer does). Requires: the
existing `build_verifier_messages` with the label made optional. Cost: K calls per note; with the guideline served from
the prompt cache each call is ≈ 0.1 G + 130 input-price-equivalent tokens plus ≈ 6 output tokens (≈ 310
input-equivalent per call; G + 130 input without caching) — ≈ 9 × F1 with caching, ≈ 33 × an uncached F1 without, and
80 requests per note (D1n ≈ 9 × on 1k-token notes, rising with N because the cached note is re-read per candidate).
Risks: the snippet loses document-level cues; per-request rate limits dominate; cache entries expire when calls are spaced
more than 5 min apart. Expected: the most accurate single-candidate verdict (no attention dilution) but unaffordable as a
primary detector on cost and on the 80-fold request volume (the batch API removes the wall-clock constraint for
evaluation runs, so hours are not the argument); its roles are the reference decision function and the H3 verifier. **P2**

**D2 — Batched verdicts over marked candidates (one call per note).** Numbered markers are inserted around every
candidate (`⟦12|03/04/2019⟧`), the guideline stays the system prompt, and the model returns one verdict line per index in
the compact format `12 DATE` / `13 -` (≈ 6 tokens each); offsets come from the marker table, never from the output.
Overlapping and nested candidates are the norm (`Dr. John Smith` ⊃ `John Smith` ⊃ `Smith`; an address line covering
STREET / CITY / STATE / ZIP), so a **de-overlap policy** is part of the design: an overlapping group is rendered as one
maximal marker whose parts are listed in the marker table (`⟦12|123 Main St, Springfield, IL 62704⟧` with parts
12a–12d), a verdict may select or split parts (`12 SPLIT 12a STREET 12b CITY 12c STATE 12d ZIP`), extend edits are
whole-token (`12 DATE +1 0`), and proposer boundary normalization (strip titles and suffixes per the i2b2 convention) is
applied before strict R_C is computed. Variants: **D2p** positives-only output — the cheapest, but it re-admits omission
because an unmentioned candidate silently defaults to "not PHI" (terminator and paired test above); **D2b** verdicts plus
a free-text `ADD` list for PHI no marker covers (an enumeration channel for the residual, aligned like F1) — closes the
candidate ceiling at small cost; **D2t** out-of-band rendering — the note is unmodified and candidates are listed after it
as (index, text, offset, ±10-character context), the control for marker bias. Targets: omission (a recognition task over
every candidate), grounding (by construction), H5's output cost (verdicts instead of the full note). Requires: proposer,
marker renderer with the de-overlap policy, parser; a `decision` detector type in the runner. Cost: one call; input =
note + ≈ 6 tokens per marker; output ≈ 6 tokens per candidate → ≈ 1.5 × F1 at the default workload (K = 80), ≈ 0.8 ×
for D2p; D2 undercuts F3 (2.8 × F1) only while K × verdict tokens stays small — K ≤ 172 at 6-token verdict lines,
K ≤ 110 at 10 tokens, K ≤ 72 with the 16-token JSON verdicts of Appendix B — which is why the compact line format is the
primary output and JSON the extended variant. Risks: marker clutter on long notes (sweep K; drop rule-negative candidates
such as section headers and lexicon drug names); attention dilution and position effects above K ≈ 150 (segment as in
F4; E9 reports accuracy by marker-position decile and by K); marker bias (E9's decoy control: known-negative candidates —
lexicon drug and condition terms, section headers, lab values — injected at matched density, with the false-accept rate
vs K as the bias estimate); verdict-list drift — missing or duplicated indices are parse failures. Expected: recall ≥ F3
at ≈ 1.5 × F1 (D2); 0.8 × only if E9 shows no D2p gap (decision-log item 7); the structured-output form of H5 and the
**primary design of the family**. **P1**

**D3 — Segment gate, then verdicts on flagged segments (two-stage decision cascade).** Stage 1 decides per segment
(line / paragraph / sentence) whether it contains PHI and which labels: a cheap LLM over the numbered segments (short
system prompt padded above the 512-token cacheable minimum; output = flagged indices) or, **D3c**, a small sentence
classifier trained on silver verdicts. Stage 2 runs D2 over the flagged segments only (φ ≈ 0.3–0.5 of the text, holding
≈ 0.6 of the candidates, where PHI concentrates in headers, signature blocks and a few sentences). Targets: strong-model
cost; routing by content rather than by TC uncertainty (complements H7 and works with no TC at all). Requires:
`segment_text`; gate prompt or classifier; a Haiku-class alias for the gate (`llm-cheap`, added to
`configs/litellm.models.yaml`; decision-log item 2). Cost: ≈ 0.9 × F1 with the gate on the Haiku-class alias; ≈ 1.35 × if
the gate runs on the Sonnet-class `llm-fast` alias (the gate then costs as much as the verdicts it saves). Risks: a gate
miss is a silent miss (report recall on unflagged segments, as for H7); φ is corpus-dependent — on dense notes the gate
saves nothing. Expected: a recall–cost curve over the gate threshold that dominates H7 when TC uncertainty is
uninformative. **P2**

**D4 — Single-token answer scored by log-probability (calibrated decision model).** The D1 prompt, but the answer is a
single token (`yes`/`no`, or a label letter) and the decision score is its log-probability; the threshold is set on dev₁
for a target recall (the recall-first protocol) instead of trusting a sampled verdict. API variant: needs `logprobs`
(OpenAI-compatible endpoints, local vLLM; not exposed by the Anthropic API) and costs like D1 with one output token.
Local variant: with prefix caching the note is prefilled once and each candidate is a short continuation, so the cost is
GPU-seconds (≈ G + N + 40 K prefill tokens per note) rather than USD — the natural deployment of the family on the A100
node (`llm-local`). Targets: calibration; recall-first thresholding; a score that H3's `verify` and H4's features can
consume. Requires: `logprobs` support in `LLMClient` (planned); a local model. Risks: verbalized yes/no probabilities of
instruction-tuned models are over-confident (temperature scaling or isotonic regression fitted on dev₁, evaluated on
dev₂/test); answer ordering (label-first vs yes/no-first) changes the scores. Expected: a tunable recall–precision curve
where D1/D2 give a single point. **P2**

**D5 — Distilled decision classifier (cross-encoder).** Train a small encoder (DeBERTa-v3-base) on D2 verdicts over the
silver pool (H9's unlabeled notes): input = snippet with the candidate marked plus the label *definition* text, output =
label or none. Hypothesis (tested with a held-out label: train on 27 definitions, evaluate on the 28th and on
INSTITUTIONAL): because the definition is an input, a new label may need only new text plus a handful of silver verdicts
(GLiNER-style extensibility) rather than a new head (H13); if the held-out-label test fails, D5 reduces to a cheaper
H13(b) head. Targets: inference cost of the whole family (≈ 0 USD, milliseconds per note on a GPU); the no-data regime
(silver verdicts are cheaper and more precise than silver spans because the candidate is fixed). Requires: proposer; one
D2 pass over the pool (one-off ≈ D2 cost × pool size, ≈ 40 USD per 5k notes at the default workload) plus GPU time.
Risks: inherits D2's omissions on the pool (iterate through D2b); a proposer change is a distribution shift. Expected:
the decision analogue of H9 — matches D2 on frequent labels, trails on the tail until silver verdicts cover it. **P3**
(P2 if H9 proves out)

**D6 — Entity-level verdicts (one verdict per distinct string).** Group candidate occurrences by normalized string
(case, whitespace, punctuation; name variants by the name parsing of S1), render every occurrence with a group and an
occurrence index (≈ 8 marker tokens), and ask for one verdict per group with an explicit exception channel
(`7 PATIENT except 7c`) for strings whose role varies by context (`May` the month vs the name; `Jordan` the country vs
the person). Targets: the "omitted repeated mention" failure directly (one decision propagates to every occurrence);
output cost by the factor 1 − 1/rep. Requires: D2 plus grouping; rep from D0 (1.6 is a placeholder — names repeat,
dates and identifiers mostly do not, so rep over *candidates* is likely below rep over gold names). Risks: wrong
propagation on role-ambiguous strings (measure exception usage and errors); group-index clutter. Expected: no recall
loss vs D2 (TOST ± 1 point), tested per mention form (full / partial / initials / possessive); at rep = 1.6, 37.5 %
fewer output tokens and ≈ 18 % lower cost (1.24 × vs 1.52 × F1). **P1** (as a D2 variant)

**D7 — Over-generate, then decide (two-model cascade).** A cheap model (or GLiNER, H16) runs an extraction pass with a
recall-maximizing instruction ("list anything that could be PHI; precision does not matter"), producing K' ≈ 1.5 P
candidates; the strong model runs D2 over them. Targets: K (the strong model reads ≈ 30 markers instead of 80). Palacios
2026 found that LLM self-verification *and* cross-model two-pass pipelines (Sonnet scrubber → Opus auditor) added
nothing, so differing models is not by itself a reason to expect a gain; what D7 adds, and what E9 tests against D2 on
the same strong model, is the asymmetric instruction — an explicitly recall-maximizing proposer and a judge whose job is
precision over an intentionally noisy set. Requires: two aliases; the proposer's R_C from D0. Cost: ≈ 0.9 × F1 with the
proposer on the Haiku-class alias; ≈ 2.1 × on the Sonnet-class `llm-fast` alias. Risks: the proposer's omissions are
unrecoverable (the ceiling is the cheap model's recall; a union with the surface-form proposer restores it at higher K);
correlated errors when proposer and judge share a provider. Expected: the best cost–recall point when a surface-form
proposer is too noisy for the note type (narrative-heavy notes with many capitalized clinical terms). **P2**

**D8 — Numbered-token labelling (proposer-free control).** Every whitespace token of the note is numbered; the model
returns `index LABEL` lines for PHI tokens only; spans are merged over consecutive indices. R_C = 1 by construction, so
D8 separates the candidate ceiling from decision quality for the whole family. Cost: input ≈ 3.5 N (indices), output
≈ 5 tokens per PHI token → ≈ 1.9 × F1. Risks: index clutter; sub-token boundaries (dates and identifiers glued to
punctuation) need a tokenizer-aware merge. **P2** (reference condition)

**Variants and exclusions.** *Vote-k verdicts* (vote-2 / vote-1 over E9's three runs) are reported as a secondary
analysis — Palacios' voting null was for extraction, and here it costs nothing extra. *D2 over redacted output* is the
H8 auditor and is run in E2. *Label-conditioned calls* (one call per label group, the MRC-NER framing of
`literature.md` §5b) are excluded on cost (× 7 categories). *Same-model extraction → decision* (the strong model
verifying its own F1 spans) is excluded on Palacios' self-verification null; D7 differs in model and in instruction.

**Relationship to the hybrids.** D2/D6 are drop-in replacements for the LLM in H1–H4 (a verdict set is a span set with
scores) and for the editor in H5; proposer (b) turns D2 into "TC-first, LLM decides" — the cascade H5 reaches for, with a
bounded output — and E9 pairs that configuration against H5 on the same TC spans, both reporting the TC false-positive
removal rate (H5's anchoring risk); D4 scores feed H3's verifier and H4's features; D3's gate is an alternative router
to H7. **What the family cannot do**: it does not help where the proposer has no candidate — lowercase or unusual-script
names, free-text PROFESSION, HOSPITAL abbreviations, LOCATION-OTHER, partial and relative dates, department and building
codes — exactly the rows where the extraction LLM is strong (§1.3). D2b's additions channel, D8, and a union with a cheap
extraction run are the mitigations; E9 measures whether the residual is worth them.

**Cost comparison.** Appendix D gives the closed-form per-note token and cost model (`note_deid.llm.costmodel`) for
F1/F3/H5 and D1–D8 under stated assumptions (1k-token note, 1.5k-token guideline, 20 PHI mentions, 80 candidates,
6-token verdict lines, Sonnet-5 price point for the strong role, Haiku-5.5 price point for the cheap role). D1 ≈ 9 × F1
(≈ 33 × an uncached F1 when nothing is cached, and 80 requests per note), D8 ≈ 1.9 ×, D2 ≈ 1.5 ×, D6 ≈ 1.2 ×,
D7 ≈ 0.9 ×, D3 ≈ 0.9 ×, D2p ≈ 0.8 ×; F3 and H5 sit at ≈ 2.8 × because they re-emit the note. D2 spans 1.5–4.3 × F1 over
the K range D0 predicts for the recommended union proposer (80–120) and verdict sizes of 6–16 tokens, so output tokens,
not input, decide the ranking among the one-call designs and the verdict format is an experimental variable. Under the
model's linear scaling the ratios are preserved when P and K scale with N; they move with PHI density (P/N), candidate
density (K/N) and verdict size — at N = 3,000 with half the default density F3 rises to 3.7 × and D2 falls to 1.4 × F1 —
so Appendix D is re-run with logged densities per note type, and all per-token constants are replaced by provider token
counts of rendered samples before any number enters the paper.

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
| D0 Candidate coverage | decision | (analysis: recall ceiling of D, by insertion mode) | none | 0 | P1 |
| D1 Per-candidate verdicts (+D1n) | decision | reference decision function; grounding | none | K (≈ 80, short) | P2 |
| D2 Batched verdicts (+D2p/D2b/D2t) | decision | omission, grounding, H5 output cost | none | 1 (bounded output) | P1 |
| D3 Segment gate → D2 | decision cascade | cost, routing without a TC | none (D3c: silver) | 1 cheap + 1 partial | P2 |
| D4 Logprob scoring | decision | calibration, recall-first thresholds | small dev | K (local: GPU-s) | P2 |
| D5 Distilled decision classifier | decision / training | inference cost, no-data regime | none (silver verdicts) | 0 (one-off D2 pass) | P3 |
| D6 Entity-level verdicts | decision | repeated-mention omissions, output cost | none | 1 | P1 |
| D7 Over-generate → decide | decision cascade | K; replication of the two-pass null with an asymmetric instruction | none | 1 cheap + 1 | P2 |
| D8 Numbered-token labelling | decision (control) | candidate ceiling (R_C = 1) | none | 1 | P2 |
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
| **RQ9** Does decision-style prompting over explicit candidates remove the LLM's omission and grounding failures, and at what cost? | Union proposer relaxed R_C ≥ 0.98 on template insertions (by construction) and ≥ 0.95 on contextual insertions, residual-text R_C reported without a prediction; D2 recall ≥ F3 recall with zero alignment failures on marker-derived spans, at a cost below F3 for K ≤ 170 with 6-token verdicts (Appendix D; the recall–cost point is reported, not thresholded); D6 = D2 in recall (TOST ± 1 point) at lower cost; D1 ≈ D2 in decision quality at ≥ 4 × cost and 80 × the requests, and D1n − D1 attributes the context effect; D4 scores scaled on dev₁ reach ECE < 0.05 (15 equal-mass bins) on dev₂/test with AUROC ≥ 0.95 and a Brier score below the D1 verbalized-confidence baseline, and the risk–coverage curve attains recall ≥ 0.99 at coverage ≥ 80 % of candidates; D7 ≈ D2 only where the cheap proposer's R_C ≥ 0.97; D8 bounds what any proposer can add | **E9**: on dev — D0 for proposers (a)–(e) by insertion mode and mention form; D1 and D1n on a 200-note subset; D2 verdict-format arms (verbose / positives-only / entity-grouped / out-of-band D2t), K sweep, decoy control, D3 and D3c gate-threshold sweeps, D4 calibration fit, D8; selection frozen (proposer, format, K, threshold, scaling). On test — the frozen D2, D6, D2p, D2b, D3c, D7 and D8 configurations, 3 runs, paired against F1/F3 from E8 and against matched-cost extraction references (F1 k-run union with k chosen so cost ≈ D2) on the same notes; D2 with proposer = TC spans paired against H5 on the same spans (both reporting the TC false-positive removal rate); vote-k verdicts as a secondary analysis | R_C per label / bucket / mention form / insertion mode and the boundary-recoverable share; rep; decision P/R over candidates; end-to-end recall decomposition (candidate miss / decision miss / label error / boundary / alignment); accuracy by marker-position decile and by K; false-accept rate on decoys; calibration (ECE, AUROC, Brier, risk–coverage); $/1k notes, calls/note, recall per USD; paired bootstrap, Holm, TOST |

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
   (admission < discharge; DOB consistent with age); the possessive, misspelled, lowercase / dictation-style and
   non-`en_US` name forms are also what the D0 coverage probe of §3.6 needs, and are not yet implemented in v0.
4. **Frequency profiles** (config): `i2b2like` (subtype proportions approximating the i2b2 2014 training distribution:
   DATE ≫ DOCTOR > HOSPITAL > PATIENT > AGE > MEDICALRECORD > CITY/STATE/PHONE/STREET/ZIP/PROFESSION/IDNUM > ORGANIZATION >
   USERNAME > COUNTRY, with FAX, EMAIL, URL, DEVICE, BIOID, HEALTHPLAN, IPADDR, SSN, VEHICLE, LICENSE, ACCOUNT, LOCATION-OTHER at
   ≤ 10 instances — exact counts to be copied from the corpus paper when the DUA copy is available); `uniform`; `tailstress`
   (tail labels up-weighted, evaluation only); `decoupled` (train profile ≠ test profile, to test rare-label generalization);
   `template-heldout` (evaluation only: header, signature and sentence-bank layouts written *after* the surface-form
   candidate proposer of §3.6 is frozen, with the proposer's source hash recorded, so that candidate coverage is not
   inflated by shared templates).
5. **Gold construction**: offsets by construction; assertion that every injected string is labeled at every occurrence;
   pre-existing residual PHI-like strings resolved before injection (step 1).
6. **Validation of realism**: human spot-check of 100 notes (naturalness of insertion, 5-point scale); a discriminator test
   (can a classifier tell LLM-inserted sentences from original ones?); detector-equivalence check against a real corpus when
   Tier A becomes available; results on real data are the ultimate check and are planned in RQ6.
7. **Splits & versioning**: split by source document (no leakage), stratified by note type; fixed seeds; `synphi-v0.1`
   with a manifest (source ids, generator config, seed, profile) so the corpus can be regenerated exactly; distribution as
   generator + manifest (and as text where source licenses permit; Asclepius is CC-BY-NC-SA).
8. **Stated limitations**: distribution shift from real notes; injected PHI is not "organically" embedded; no real
   institution-specific idiosyncrasies (simulated via the lexicon); the surface-form candidate proposer (§3.6) and the
   injection engine share regex/template provenance, so proposer coverage on template insertions is an upper bound.

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
  and the cost log must show that it was: a configuration whose logged `cache_read_tokens` is zero on the second call is
  re-run after the prefix is fixed. Cache writes (1.25 ×) are not budgeted because the serialized runner keeps the prefix
  warm within the 5-minute TTL. Cache reads also do not count toward input-token rate limits on the Anthropic API,
  which matters at concurrency 1.
- **Log-probabilities** (D4) are only available on OpenAI-compatible endpoints and the local vLLM server (`llm-local`);
  the client is to expose them as an optional `logprobs` request parameter (not implemented yet) and leave the score
  `None` elsewhere.
- `note_deid.llm.costmodel`: closed-form per-note token/cost estimates for F1/F3/H5 and D1–D8 (Appendix D).

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
| **P2 — inference-time hybrids** | H1, H2, H3, H5, H7, H8 → **E2, E5**; D0, D2 (+D2p/D2b/D2t), D6, D8, D7, then D1/D1n/D3/D4 → **E9** | P1 |
| **P3 — training-time hybrids** | H9, H10, H13 (+ H11, H12, D5 if budget) → **E3, E4** | P1, A100 sweeps |
| **P4 — surrogation** | S1–S3 → **E7** | P2 |
| **P5 — robustness & paper** | E6 (cross-source, note type, institutional; i2b2 transfer if access), gold adjudication, ablations, writing | P2–P4 |

Compute/cost envelope (to be measured in P1): ≈ 2k test notes × ~12 LLM configurations × 3 runs ≈ 70k calls; with caching
and dev-subset tuning the paid volume should stay well below that; all LLM runs are cached and resumable. E9 adds
≈ 16k short calls per run for D1/D1n on the 200-note dev subset and about eight further one- or two-call configurations on
dev plus the frozen set on test (Appendix D).

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
  outputs); gold gaps; English only; single taxonomy; shared regex/template provenance between the candidate proposer of
  §3.6 and the injection engine (mitigated by the `template-heldout` profile and by reporting coverage per insertion mode).

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
   different providers, pinned by dated identifiers in `configs/litellm.models.yaml`; a third, Haiku-class `llm-cheap`
   alias (added 2026-11-02 as a placeholder) is what the D3 gate and the D7 proposer are costed on — on the Sonnet-class
   `llm-fast` alias those cascades cost more than F1 (Appendix D).
3. Human annotation budget for H11/H12 — recommended ≥ 100 notes adjudicated (or none: simulate with gold).
4. Which prior artifacts exist and in what format (guideline version, annotation format, TC checkpoint) — Appendix A.
5. Target venue and timeline.
6. Candidate proposer for the decision-model family (§3.6) — recommended **union of surface-form regex/capitalization
   rules and TC spans** (proposer (e) without the LLM member), with D0 coverage reported per label before any D run.
7. Verdict output format for D2 — recommended **one compact verdict line per candidate** (`12 DATE` / `13 -`) as the
   primary condition, JSON verdicts as the extended variant (boundary edits, splits), positives-only (D2p) and
   entity-grouped (D6) as the cost arms; positives-only becomes the default only if E9 shows no recall gap.
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
- **Decision, per candidate (D1/D1n/D4)**: the verifier prompt applied to every proposed candidate, with the proposed label
  optional (`null` for shape-less candidates; the model returns the label); D1n puts the whole note into the cached system
  prefix; D4 asks for a single answer token (`yes`/`no`, or a label letter) and reads its log-probability as the score.
- **Decision, batched (D2)**: input = the note with numbered markers around every candidate (`⟦12|03/04/2019⟧`);
  overlapping candidates form one maximal marker whose parts (12a–12d) are listed in the marker table; output = one line
  per index, `12 DATE` / `13 -`, with optional whole-token extends (`12 DATE +1 0`) and splits
  (`12 SPLIT 12a STREET 12b CITY 12c STATE 12d ZIP`); the JSON form
  `{"decisions": [{"i": 12, "phi": true, "label": "DATE", "extend": [0, 0], "split": [...]}]}` is the extended variant
  (≈ 16 tokens per verdict instead of ≈ 6). **D2p** = accepted candidates only, terminated by `END <n>`; **D2b** = D2 plus
  `ADD <text> <LABEL>` lines for PHI without a marker (aligned like F1); **D2t** = unmodified note followed by a candidate
  table (index, text, offset, ±10-character context); **D6** = one index per distinct string with occurrence suffixes and
  `except` lists (`7 PATIENT except 7c`); **D8** = every whitespace token numbered, output `index LABEL` for PHI tokens.
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
note, and every per-token constant is replaced by the provider token count of a rendered sample before any number enters
the paper. Prices are as given by the bundled Claude API reference (model table cached 2026-10-06, read 2026-11-02), not
by the live pricing page: the "strong" role at the Sonnet-5 point (2 / 10 USD per 1M input / output tokens — the model
behind the `llm-fast` alias in `configs/litellm.models.yaml`; the OpenAI model behind `llm-strong` was not re-priced in
this session and is supplied with `--input-per-m/--output-per-m`), the "cheap" role (D3 gate, D7 proposer) at the
Haiku-5.5 point (0.10 / 0.50, valid for prompts ≤ 100K tokens; the `llm-cheap` alias); cache reads at 0.1 × the input
price; batch API 0.5 ×. Cache writes (1.25 ×) are omitted because the serialized runner keeps the prefix warm within the
5-minute TTL. Like the model identifiers themselves these prices are placeholders to re-verify. The hours column is a
lower bound at concurrency 1 (1 s request overhead plus decoding at 60 tokens/s; no prefill, no retries, no back-off).
USD columns exclude local compute, which the "local compute" column names: surface-form proposer rules are negligible;
TC-based and GLiNER proposers, the D3c gate classifier and the D5 encoder add one local pass per note (GPU-seconds,
logged as for S_TC).

| Symbol | Default | Meaning |
|---|---:|---|
| N | 1,000 | note tokens |
| G | 1,500 | system prompt: guideline + format instructions (the cacheable prefix) |
| gate prompt | 600 | D3 gate system prompt, kept above the 512-token cacheable minimum |
| P | 20 | gold PHI mentions per note |
| K | 80 | candidates from the surface-form proposer |
| over-generation | 1.5 × P | D7 proposer output, which defines K' (= 30) |
| S, φ, φ_K | 25, 0.4, 0.6 | gate segments per note, fraction of the text flagged, fraction of the candidates inside flagged segments (D3) |
| rep | 1.6 | occurrences per distinct candidate string (D6; measured by D0) |
| additions | 0.1 × P | D2b free-text additions |
| per-candidate tokens | 130 / 20 / 6 / 8 | snippet + framing (D1, D4) / user turn with the note in the cached prefix (D1n) / inline marker (D2) / group + occurrence marker (D6) |
| per-verdict tokens | 6 / 5 | compact verdict line `12 DATE` (D2, D6, D1) / positives-only line (D2p); JSON verdicts ≈ 16 |
| per-span tokens | 15 / 12 | F1 JSON item / inline tag pair (F3, H5) |
| D8 | 2.5 / 2.5 / 5 | extra input tokens per note token for numbering / PHI tokens per mention / output tokens per labelled token |

**Default workload** (USD per 1,000 notes; "cache" = guideline served from the prompt cache; "x F1" on the cached column):

| ID | Design | local compute | calls/note | input (uncached) | input (cache read) | output | USD/1,000 no cache | USD/1,000 cache | xF1 | USD/1,000 cache+batch | h/1,000 @1 (lb) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | extraction, JSON strings (reference) | none | 1.0 | 1,000 | 1,500 | 300 | 8.00 | 5.30 | 1.00 | 2.65 | 1.7 |
| F3 | extraction, inline tags | none | 1.0 | 1,000 | 1,500 | 1,240 | 17.40 | 14.70 | 2.77 | 7.35 | 6.0 |
| H5 | TC-first editor, full tagged text out | TC pass | 1.0 | 1,240 | 1,500 | 1,240 | 17.88 | 15.18 | 2.86 | 7.59 | 6.0 |
| D1 | one verdict call per candidate, snippet context | proposer | 80 | 10,400 | 120,000 | 480 | 265.60 | 49.60 | 9.36 | 24.80 | 24.4 |
| D1n | one verdict call per candidate, note in cached prefix | proposer | 80 | 1,600 | 200,000 | 480 | 408.00 | 48.00 | 9.06 | 24.00 | 24.4 |
| D2 | batched verdicts over marked candidates | proposer | 1.0 | 1,480 | 1,500 | 480 | 10.76 | 8.06 | 1.52 | 4.03 | 2.5 |
| D2p | D2, positives-only output | proposer | 1.0 | 1,480 | 1,500 | 100 | 6.96 | 4.26 | 0.80 | 2.13 | 0.7 |
| D2b | D2 + free-text additions for unproposed PHI | proposer | 1.0 | 1,480 | 1,500 | 510 | 11.06 | 8.36 | 1.58 | 4.18 | 2.6 |
| D3 | cheap-model segment gate, then D2 on flagged | proposer | 2.0 | 1,788 | 2,100 | 318 | 7.44 | 4.69 | 0.88 | 2.34 | 2.0 |
| D3c | classifier segment gate, then D2 on flagged | proposer + gate classifier | 1.0 | 688 | 1,500 | 288 | 7.26 | 4.56 | 0.86 | 2.28 | 1.6 |
| D4 | single-token answer scored by logprob (API variant) | proposer | 80 | 10,400 | 120,000 | 80 | 261.60 | 45.60 | 8.60 | 22.80 | 22.6 |
| D5 | distilled decision classifier (inference) | proposer + encoder | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 |
| D6 | D2 with one verdict per distinct string | proposer | 1.0 | 1,640 | 1,500 | 300 | 9.28 | 6.58 | 1.24 | 3.29 | 1.7 |
| D7 | cheap over-generation, then strong D2 | none | 2.0 | 2,180 | 3,000 | 630 | 7.63 | 4.80 | 0.91 | 2.40 | 3.5 |
| D8 | numbered-token labelling (proposer-free control) | none | 1.0 | 3,500 | 1,500 | 250 | 12.50 | 9.80 | 1.85 | 4.90 | 1.4 |

**Sensitivity.** D2's cost is output-bound: with K = 120 (the upper end of what D0 predicts for the union proposer) it
is 2.06 × F1 at 6-token verdict lines and 2.97 × at 10 tokens; with the 16-token JSON verdicts of Appendix B it is
3.03 × at K = 80 and 4.33 × at K = 120. D2 undercuts F3 (2.77 ×) only while K × verdict tokens stays small: K ≤ 172 at
6 tokens, K ≤ 110 at 10, K ≤ 72 at 16. **Long notes** (N = 3,000, P = 60, K = 240, S = 75; same prices): the ranking
is unchanged and the ratios move by a few percent because every N- or K-dependent term scales together and the fixed
guideline term is small — F1 15.30, F3 43.50, H5 44.94, D1 148.80, D1n 240.00, D2 23.58, D2p 12.18, D2b 24.48, D3 13.45,
D3c 13.07, D4 136.80, D6 19.14, D7 13.77, D8 28.80 USD per 1,000 notes with caching; D1n is the exception, because its
cached note prefix is re-read once per candidate. With **half the PHI and candidate density** at N = 3,000 (P = 30,
K = 120) the ratios shift — F3 3.69 ×, D2 1.38 ×, D6 1.18 ×, D2p 0.86 × — so the table is re-run with logged densities
per note type. With the **cheap role priced like the current `llm-fast` alias** (Sonnet-5), D3 rises to 1.35 × and D7 to
2.12 × F1: the cascades beat F1 only with a Haiku-class gate or proposer. At the **Opus-5 price point** for the strong
role every strong-only figure is 2.5 × higher and its ratio unchanged; D3 and D7 move to 0.87 × because their cheap-model
term does not scale.

**Reading the table.**

- Among the one-call designs the *output* term decides the order: D2's 480 verdict tokens cost more than F1's 300
  span-item tokens; D2p (positives only) and D6 (one verdict per string) recover most of that; F3 and H5 pay ≈ N output
  tokens to re-emit the note; D8 pays ≈ 2.5 N input tokens for the indices. The verdict format is therefore an
  experimental variable in E9, not an implementation detail.
- The per-candidate designs (D1, D1n, D4) cost ≈ 9 × F1 with caching (≈ 33 × an uncached F1 without, because the
  guideline is re-read 80 times); at concurrency 1 they also take about a day per 1k notes, but the batch API removes the
  wall-clock constraint for evaluation runs, so their exclusion as primary detectors rests on cost and on the 80-fold
  request volume, not on latency.
- Gating (D3/D3c) and over-generation (D7) bring the strong-model cost to ≈ 0.9 × F1 while keeping D2's grounding
  properties, provided the gate or proposer runs on a Haiku-class model; their risk is the recall of what the gate or
  proposer does not pass on, which E9 reports separately. The D3 rows assume that 60 % of the candidates sit in the 40 %
  of text flagged; if candidates are spread uniformly the strong stage is cheaper still, if PHI is denser in the flagged
  part it is dearer.
- D5 moves the whole family to a local encoder at inference (≈ 0 USD, GPU time logged); its one-off cost is one D2 pass
  over the silver pool, ≈ 40 USD per 5k notes at the default workload, plus training time.
- Batch processing halves every figure and suits the offline evaluation runs; caching needs calls that share the prefix
  within the cache lifetime (5 min by default), which the serialized runner guarantees.
