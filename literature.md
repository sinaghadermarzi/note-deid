# Literature review: hybrid LLM + token-classifier de-identification of clinical notes

Compiled 2026-09-14 for the `note-deid` project. Purpose: ground the framework design (see `framework-designs.md`)
and provide the related-work backbone of the planned paper.

**Verification legend** (every entry carries one):

- **(V)** primary source opened this session (abstract, page or README fetched); numbers quoted from it.
- **(K)** standard reference from prior knowledge, not re-fetched this session. Bibliographic details are believed
  accurate but must be re-checked against the publisher page before the paper is submitted.
- **(S)** seen only through a secondary source (search snippet, blog, another paper). Treat quantitative claims as provisional.

---

## 1. Task definition, taxonomies, and shared tasks

**HIPAA Safe Harbor (45 CFR §164.514(b)(2))** (K). Eighteen identifier classes: names; geographic subdivisions smaller
than a state; all elements of dates (except year) directly related to an individual, and ages over 89; telephone; fax;
email; SSN; medical record numbers; health-plan beneficiary numbers; account numbers; certificate/license numbers; vehicle
identifiers; device identifiers; URLs; IP addresses; biometric identifiers; full-face photographs; and any other unique
identifying number, characteristic or code. "Expert Determination" is the alternative pathway (statistical re-identification
risk). Clinical de-identification research uses Safe Harbor as the *minimum* target and usually adds categories
(professions, hospital names, all ages, all dates) because they act as quasi-identifiers.

**i2b2 2006 de-identification challenge** — Uzuner, Luo, Szolovits. *Evaluating the state-of-the-art in automatic
de-identification.* JAMIA 2007;14(5):550–563 (K). 889 discharge summaries, eight PHI classes (patients, doctors, hospitals,
IDs, dates, locations, phone numbers, ages). Established token/entity-level P/R/F1 reporting.

**i2b2/UTHealth 2014, Track 1** — Stubbs, Kotfila, Uzuner. *Automated systems for the de-identification of longitudinal
clinical narratives: overview of 2014 i2b2/UTHealth shared task Track 1.* JBI 2015;58(Suppl):S11–S19 (K); corpus paper
Stubbs & Uzuner, *Annotating longitudinal clinical narratives for de-identification: the 2014 i2b2/UTHealth corpus.* JBI
2015;58(Suppl):S20–S29 (K). 1,304 longitudinal records from 296 diabetic patients; taxonomy of **7 categories with 25 sub-categories — 28 leaf labels** once the undivided PROFESSION, AGE and DATE are counted
(NAME{PATIENT, DOCTOR, USERNAME}, PROFESSION, LOCATION{HOSPITAL, ORGANIZATION, STREET, CITY, STATE, COUNTRY, ZIP,
LOCATION-OTHER}, AGE, DATE, CONTACT{PHONE, FAX, EMAIL, URL, IPADDR}, ID{SSN, MEDICALRECORD, HEALTHPLAN, ACCOUNT, LICENSE,
VEHICLE, DEVICE, BIOID, IDNUM}). Evaluation script reports strict (exact offsets + type) and relaxed (overlap) entity-level
P/R/F1, micro and per category, plus a HIPAA-only subset. Several subtypes are extremely rare in the training data
(e.g., FAX, EMAIL, URL, IPADDR, SSN, HEALTHPLAN, VEHICLE, DEVICE, BIOID, COUNTRY) — this **long tail is the concrete
instance of the "rare label" failure mode** we study. The de-facto standard benchmark since 2015 and the taxonomy this
project adopts as its target label set (`src/note_deid/labels.py`).

**CEGS N-GRID 2016, Track 1** — Stubbs, Filannino, Uzuner. *De-identification of psychiatric intake records: overview of
2016 CEGS N-GRID shared tasks Track 1.* JBI 2017;75(Suppl):S4–S18 (K). 1,000 psychiatric intake records; same taxonomy
family; showed domain shift from the 2014 data.

**Text Anonymization Benchmark (TAB)** — Pilán, Lison, Øvrelid, Papadopoulou, Sánchez, Batet. *The Text Anonymization
Benchmark (TAB): a dedicated corpus and evaluation framework for text anonymization.* Computational Linguistics
2022;48(4):1053–1101 (K). Distinguishes **direct identifiers** from **quasi-identifiers** (identifying only in
combination), annotates "need to mask" per span, and proposes privacy-oriented recall (weighted by identifier type) plus
token-level precision as a utility proxy. Also: Lison, Pilán, Sánchez, Batet, Øvrelid. *Anonymisation models for text data:
state of the art, challenges and future directions.* ACL 2021 (K). Relevance: our evaluation adopts the
recall-first / utility-second framing and the quasi-identifier notion.

**Reviews.** Meystre, Friedlin, South, Shen, Samore. *Automatic de-identification of textual documents in the electronic
health record: a review of recent research.* BMC MIDM 2010;10:70 (K). Kovačević, Bašaragin, Milošević, Nenadić.
*De-identification of clinical free text using natural language processing: a systematic review of current approaches.*
Artif Intell Med 2024;151:102845 (K) — documents the shift to transformer models, the dominance of i2b2 2014 as the sole
benchmark, and the scarcity of external validation. *A survey on current trends and recent advances in text anonymization.*
arXiv:2508.21587 (2025) (S) — broader anonymization survey including LLM-era methods.

---

## 2. Rule-based, statistical and neural systems (pre-LLM), including early hybrids

| System / paper | Approach | Notes |
|---|---|---|
| **deid** — Neamatullah et al., *Automated de-identification of free-text medical records.* BMC MIDM 2008;8:32 (K) | dictionaries + regex + heuristics | Shipped with a gold-standard corpus of 2,434 MIMIC-II nursing notes (PhysioNet `deid` 1.1). The corpus now requires PhysioNet credentialed access (V, physionet.org/content/deid/1.1). |
| **MIST** — Aberdeen et al., *The MITRE Identification Scrubber Toolkit: design, training, and assessment.* IJMI 2010;79(12):849–859 (K) | CRF sequence labeling + "resynthesis" (surrogates) | Early integrated detection + surrogation toolkit. |
| **BoB (best-of-breed)** — Ferrández, South, Shen, Friedlin, Samore, Meystre. JAMIA 2013;20(1):77–83 (K) | rules + SVM/CRF, two-stage (recall-oriented then precision-oriented) | The canonical *rule + ML* hybrid for VA notes; the "recall stage then precision stage" cascade is a direct ancestor of our cascade designs. |
| **Yang & Garibaldi**, *Automatic detection of protected health information from clinic narratives.* JBI 2015;58(Suppl):S30–S38 (K) | CRF + rules + post-processing | Best system of the 2014 challenge (strict F1 ≈ 0.94). |
| **Liu, Tang, Wang, Chen**, *De-identification of clinical notes via recurrent neural network and conditional random field.* JBI 2017;75(Suppl):S34–S42 (K) | ensemble of CRF, Bi-LSTM-CRF and rules | Top of N-GRID 2016; evidence that heterogeneous *classical* ensembles help. |
| **NeuroNER** — Dernoncourt, Lee, Uzuner, Szolovits. *De-identification of patient notes with recurrent neural networks.* JAMIA 2017;24(3):596–606 (K) | char + word Bi-LSTM-CRF | First neural de-id at state-of-the-art level on i2b2 2014. |
| **Kim, Heider, Meystre**, *Ensemble-based methods to improve de-identification of electronic health record narratives.* AMIA Annu Symp Proc 2018 (K) | voting / stacking across de-id systems | Ensembles raise recall at some precision cost — the same trade-off we expect from union fusion. |
| **Yang, Lyu, Li, Lee, Bian, Hogan, Wu**, *A study of deep learning methods for de-identification of clinical notes in cross-institute settings.* BMC MIDM 2019;19(Suppl 5):232 (K) | LSTM-CRF / transformers, cross-institution | Performance drops across institutions; local fine-tuning recovers it (domain shift baseline for RQ6). |
| **Philter** — Norgeot et al., *Protected Health Information filter (Philter): accurately and securely de-identifying free-text clinical notes.* npj Digit Med 2020;3:57 (K) | rules + NLP, recall-maximizing ("safe by design") | Reports very high recall at lower precision on i2b2 2014 and UCSF notes; standard high-recall baseline. |
| **Johnson, Bulgarelli, Pollard**, *Deidentification of free-text medical records using pre-trained bidirectional transformers.* ACM CHIL 2020 (K) | BERT-family token classification | Released models; strong on i2b2 2014; MIMIC-based synthetic PHI evaluation. |
| **Murugadoss et al.**, *Building a best-in-class automated de-identification tool for electronic health records through ensemble learning.* Patterns 2021;2(6):100255 (K) | ensemble of neural NER + rules (nference) | Scale de-id of Mayo notes; ensemble > single model. |
| **Locally augmented ensembles** — *Scaling text de-identification using locally augmented ensembles.* medRxiv 2024, doi:10.1101/2024.06.20.24308896 (S; full text returned HTTP 403 this session) | ensemble + institution-specific dictionaries | Adding local dictionaries improves P and R **without retraining** the neural components; evaluated on Duke and Mayo real notes. Closest "hybrid without LLM" work. |
| **Chambon, Wu, Steinkamp, Adleberg, Cook, Langlotz**, *Automated deidentification of radiology reports combining transformer and "hide in plain sight" rule-based methods.* JAMIA 2023;30(2):318–328 (K) | transformer NER + HIPS surrogates | Detection + surrogation in one pipeline; motivates our surrogation arm. |
| **Stanford TiDE** (K/S) | pattern + NER tool | Used as a purpose-built baseline by Palacios et al. 2026 (F1 0.779 on pediatric oncology notes). |
| **Comparative transformer evaluation** — arXiv:2204.07056 (S) | BERT/RoBERTa/… on i2b2 2014 | RoBERTa-large best in that comparison. |
| **Single-center multi-specialty transformer benchmark** — PMC12719064 (2025) (S) | fine-tuned transformers | >99% accuracy, RoBERTa-large ≈ 96.7% P/R after fine-tuning (snippet). |
| **OpenMed-PII** — HF `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` (V) | DeBERTa-v3-large token classifier, **54 labels**, trained on NVIDIA Nemotron-PII (50k train / 5k val / 45k test) | Micro-F1 0.961 / P 0.969 / R 0.953 on its own test set; Apache-2.0; part of the local-first OpenMed toolkit (Python + Apple MLX). Second open TC baseline with a much richer label set than OPF. |
| **OpenAI Privacy Filter (OPF)** — github.com/openai/privacy-filter, HF `openai/privacy-filter` (V) | 1.5B-parameter MoE (50M active), gpt-oss-like encoder converted to a **bidirectional banded-attention token classifier** (window 257), 128k context, **8 labels** × BIOES = 33 classes | Labels: `private_person, private_date, private_address, private_phone, private_email, private_url, account_number, secret`. CLI `opf` for redaction, `opf eval`, and `opf train <train.jsonl>`; JSONL record `{"text", "spans": {"<label>: <value>": [[start, end]]}, "info"}` (character offsets, end exclusive). Apache-2.0. README-stated limitations: "static label policy" (new labels need fine-tuning), weaker on uncommon names, non-English text, novel secret formats; explicitly "not an anonymization guarantee". No AGE / PROFESSION / HOSPITAL / ORGANIZATION labels. |
| **Tonic.ai benchmark of OPF** (V, vendor blog 2026) | OPF vs Tonic Textual on 500+ real documents incl. EHR notes | OPF precision comparable (0.77–0.85) but **recall very low out of the box** (≈ 38% on EHR notes, 10% on web crawls); misses conversational formats, unusual email TLDs, letterhead blocks, long-tail presentation formats. Vendor-authored; treat as indicative of the *format long tail* failure mode. |
| **GLiNER** — Zaratiana, Tomeh, Holat, Charnois. *GLiNER: generalist model for named entity recognition using bidirectional transformer.* NAACL 2024 (K); PII variants (`urchade/gliner_multi_pii-v1`, NVIDIA GLiNER-PII, `fastino/gliner2-privacy-filter-PII-multi`) (S) | bi-encoder span model conditioned on natural-language label names | Zero-shot label extension without generation; a cheap "middle" component between TC and LLM (optional arm H16). |

**Take-aways for design.** (i) Every generation of de-id systems that won a shared task was a *hybrid* (rules + CRF,
rules + LSTM, ensembles), and ensembles reliably trade precision for recall. (ii) Purpose-built token classifiers are
brittle to *format* and *institution* shift and to label-set changes. (iii) Modern open token classifiers (OPF, OpenMed) are
cheap and fine-tunable but ship with fixed, non-clinical taxonomies — the AGE/PROFESSION/HOSPITAL gap must be closed by
fine-tuning or by another component.

---

## 3. LLM-based de-identification

- **DeID-GPT** — Liu et al., *DeID-GPT: zero-shot medical text de-identification by GPT-4.* arXiv:2303.11032 (2023) (V,
  abstract). GPT-4 masking prompts on i2b2 2014 and synthetic data; high accuracy, but the model rewrites text, which
  complicates span-level evaluation.
- **Altalla' et al.**, *Evaluating GPT models for clinical note de-identification.* Sci Rep 2025 (PMID 39890969) (V, abstract
  via snippet). GPT-4: **precision 0.9925, recall 0.8318, F1 0.8973** — the high-precision / recall-gap profile matches
  the "LLM misses spans" failure mode observed in our own experiments.
- **LLM-Anonymizer** — Wiest et al., *Deidentifying medical documents with local, privacy-preserving large language models:
  the LLM-Anonymizer.* NEJM AI 2025, doi:10.1056/AIdbp2400537 (medRxiv 2024.06.11.24308355) (S). Local llama.cpp inference
  with Llama-2/3 (7B–70B), Mistral 7B, Phi-3; Llama-3-70B had the lowest false-negative rate; larger models → fewer misses.
- **Palacios, Neeley, Otto, et al.**, *Institution-specific LLM prompting recovers PHI that de-identification systems and their
  gold standards both miss.* arXiv:2608.17051 (Aug 2026) (V, full text). 100 pediatric-oncology notes, 5,322 PHI spans;
  eight LLMs under three prompts vs Stanford TiDE and OpenMed-PII. Best LLM F1 0.918 vs TiDE 0.779; naming institution-specific
  missed categories in the prompt recovered 79% of previously missed PHI; final prompt recall 0.981. Output format: the LLM
  returns the full text with typed placeholders; evaluation by exact substring search (type-agnostic). **Key negative
  result:** 14 agentic configurations (dual-pass self-refinement, Sonnet scrubber → Opus auditor, cross-model dual pass,
  3-run majority voting) **did not beat single-pass** on F1 — "de-identification is a single-read task", over-redactions are
  *correlated across runs* so voting cannot remove them, and ensembles cost 3–6× more. Also: 414 LLM "false positives" were
  reviewed and found to be real PHI missed by the gold standard (re-annotation added 227 spans). TiDE failed where context is
  required (Other-Unique-ID recall 15.7%, phone 47.9%, geographic 65%) while LLM misses concentrated on institution
  abbreviations, building/facility names, department codes and partial dates.
- **Clinical text de-identification using LLMs** — Dahleh, MIT MEng thesis 2025 (S). Locally deployed pipeline reported to
  surpass GPT-4o and cloud providers on accuracy and cost.
- **Not what the doctor ordered: surveying LLM-based de-identification and quantifying clinical information loss.**
  arXiv:2509.14464, EMNLP 2025 (V, abstract). Reporting is inconsistent across studies; standard classification metrics
  fail to capture clinically significant *over-redaction*; proposes clinician-validated information-loss measurement.
- **Vendor comparison** (S; IntuitionLabs guide citing a John Snow Labs study on 48 expert-annotated documents): JSL 96% F1,
  Azure Health Data Services 91%, Amazon Comprehend Medical 83%, zero-shot GPT-4o 79%. Vendor-authored.
- **Prompt ensembles for medical NER** — arXiv:2505.08704 (S). Prompt-level ensembling for reliability.
- **Differentially private de-identification of Dutch clinical notes** — arXiv:2604.21421 (S). Peripheral (DP training).

**Take-aways.** LLMs (a) reach the highest recall on *context-dependent* and institution-specific PHI when the prompt names
the categories, (b) show a persistent recall gap in zero-shot / generic-prompt settings, (c) have errors that are
**correlated across runs and across models**, so homogeneous LLM ensembles buy little, (d) expose gold-standard gaps, and
(e) cost 1–2 orders of magnitude more than a token classifier. None of these papers pairs the LLM with a *trained* token
classifier and analyses whether their error sets are complementary — the gap this project targets.

---

## 4. LLM as teacher: distillation, silver labels, active learning

- **SHIELD** — *SHIELD: a diverse clinical note dataset and distilled small language models for enterprise-scale
  de-identification.* arXiv:2605.03301 (2026) (V, abstract). 1,381 notes / 10,229 spans / 9 PHI categories, built with
  LLM pre-annotation + human adjudication and set-cover diversity sampling; four LLM teachers distilled into a DeBERTa-v3
  student (span P 0.89 / R 0.88 micro) that **loses macro recall on rare and institution-specific categories (0.81 vs 0.90
  teacher)**; authors suggest hybrids of broad-coverage and specialized models. Stanford Medicine (Posada, Love, Datta,
  Desai; the TiDE group); nine categories AGE, DATE, DOCTOR, HOSPITAL, ID, LOCATION, PATIENT, PHONE, WEB; teachers
  Gemini 2.5 Pro/Flash, GPT-OSS-120B, Llama 4 Maverick (Gemini 2.5 Flash distilled). Released at
  github.com/susom/shield_dataset (V) — a credible external test set for RQ6.
- **LLMs-in-the-loop Part 2** — *Expert small AI models for anonymization and de-identification of PHI across multiple
  languages.* arXiv:2412.10918 (John Snow Labs, 2024) (V, abstract). Small NER models trained with LLM-in-the-loop data reach
  F1-micro 0.953–0.978 across eight languages and are claimed to beat GPT-4o.
- **Enhancing clinical models with pseudo data for de-identification.** arXiv:2506.12674 (2025) (V, abstract). Pre-training
  encoders on text with realistic pseudo-PHI substituted for redaction placeholders improves downstream de-id; generation
  code and models released.
- **UniversalNER** — Zhou et al., *UniversalNER: targeted distillation from large language models for open named entity
  recognition.* ICLR 2024, arXiv:2308.03279 (K). Distilling ChatGPT annotations into 7–13B open models with mission-focused
  instruction tuning; the student matches or beats the teacher on many NER benchmarks.
- **Gu et al.**, *Distilling large language models for biomedical knowledge extraction: a case study on adverse drug events.*
  arXiv:2307.06439 (2023) (K). PubMedBERT student trained on GPT-3.5 labels outperforms the teacher.
- **CanDist** — *Prompt candidates, then distill: a teacher-student framework for LLM-driven data annotation.* ACL 2025,
  arXiv:2506.03857 (V, snippet). LLM proposes candidate labels; the student learns to pick the right one (distribution
  refinery) — a principled way to consume noisy LLM annotations.
- **From selection to generation: a survey of LLM-based active learning.** ACL 2025, arXiv:2502.11767 (V, snippet).
  Kholodna et al., *LLMs in the loop: leveraging LLM annotations for active learning in low-resource languages.*
  arXiv:2404.02261 (S). Goel et al., *LLMs accelerate annotation for medical information extraction.* ML4H 2023,
  arXiv:2312.02296 (S). Wang et al., *Want to reduce labeling cost? GPT-3 can help.* Findings of EMNLP 2021 (K).
- **Self-consistency** — Wang et al., ICLR 2023 (K): majority voting over sampled generations; the basis for
  vote-filtered silver labels.

**Take-aways.** Distillation from LLM silver labels is established and works well for *frequent* categories; the reported
weakness is exactly the rare/institution-specific tail. No de-id study combines distillation with targeted rare-label
augmentation, disagreement-driven active learning, or a guideline co-refinement loop, nor reports learning curves versus
the number of human-labeled notes.

---

## 5. Span labeling with LLMs: output formats and grounding

- **Strategies for span labeling with large language models.** arXiv:2601.16946 (2026) (V, abstract). Three families —
  *tagging* the input, *indexing* numeric positions, *matching* span content. "Tagging remains a robust baseline"; a
  constrained-decoding method (LogitMatch) forces outputs to be valid input spans and removes matching inconsistencies.
  Generative LLMs "lack an explicit mechanism to refer to specific parts of their input".
- **Assessment of generative NER in the era of LLMs.** arXiv:2601.17898 (2026) (V, snippet). Five formats compared (inline
  bracketed, inline XML, category-grouped JSON, occurrence-based JSON, offset-based JSON); the winner depends on model and
  dataset; offset generation is unreliable ("resembling hallucinations").
- **GPT-NER** — Wang et al., arXiv:2304.10428 (2023) (K): inline `@@…##` markers + a self-verification step to cut false
  positives. **PromptNER** — Ashok & Lipton, arXiv:2305.15444 (K). **Xie et al.**, *Empirical study of zero-shot NER with
  ChatGPT.* EMNLP 2023 (K): decomposition + self-consistency improve zero-shot NER.
- **Unified biomedical NER framework with LLMs** — arXiv:2510.08902 (S): symbolic tagging beat JSON and HTML formats.
- **Evaluating NER using few-shot prompting with LLMs** — arXiv:2408.15796 (S); **label-guided ICL** — arXiv:2505.23722 (S).

**Relevance.** Our existing pipeline (JSON list of span strings, then string search) is in the *content-matching* family.
Its known failure modes — omitted mentions, non-verbatim strings, ambiguous repeated occurrences — are partly
format-induced. The framework therefore includes an output-format ablation (inline tagging, occurrence-indexed JSON,
constrained decoding for local models, fuzzy alignment) so that "LLM misses" can be decomposed into semantic misses and
grounding failures before hybrids are evaluated.

---

## 6. Annotation-guideline refinement and prompt optimization

- **Refining and reusing annotation guidelines for LLM annotation.** arXiv:2605.20809 (2026) (V, snippet). Iterative
  "moderation" framework that refines guidelines from disagreement, evaluated on NCBI-Disease, BC5CDR and BioRED with GPT,
  Gemini and DeepSeek families.
- **Efficient medical NER with limited data: enhancing LLM performance through annotation guidelines.** Int J Med Inform 2025
  (S1386505625004472) (V, snippet). Human-oriented guidelines in the prompt beat few-shot examples alone.
- **Improving LLM-based event extraction with annotation guidelines.** Frontiers in AI 2026 (S). Guidelines improve both
  boundaries and previously missed entities.
- **Guideline learning for in-context information extraction** — Pang, Cao, Ding, Luo. EMNLP 2023, ACL Anthology
  2023.emnlp-main.950 (V). Synthesizes guidelines from a few error cases and retrieves them at inference.
  Related: *Instruction-tuning LLMs for event extraction with annotation guidelines*, arXiv:2502.16377 (S).
- **Automatic prompt optimization**: ProTeGi (Pryzant et al., EMNLP 2023) (K); OPRO (Yang et al., ICLR 2024) (K); DSPy
  (Khattab et al., ICLR 2024) (K); Self-Refine (Madaan et al., NeurIPS 2023) (K).

**Relevance.** Our existing "LLM + guideline, refined from annotated notes" loop is an instance of error-driven guideline
refinement. The literature supports it but warns about dev-set overfitting; the framework formalizes it (H12) with a
held-out protocol and couples it to the token classifier's error profile.

---

## 7. Surrogation ("hiding in plain sight") and synthetic PHI

- **Carrell, Malin, Aberdeen, Bayer, Clark, Wellner, Hirschman.** *Hiding in plain sight: use of realistic surrogates to
  reduce exposure of protected health information in clinical text.* JAMIA 2013;20(2):342–348 (K). Realistic surrogates make
  residual leaked PHI hard to distinguish; ≈ 90% of residual identifiers effectively concealed at average/high PHI densities
  (S, snippet).
- **Yeniterzi, Aberdeen, Bayer, Wellner, Hirschman, Malin.** *Effects of personal identifier resynthesis on clinical text
  de-identification.* JAMIA 2010;17(2):159–168 (K). Training/evaluating de-id models on resynthesized text is feasible with
  modest degradation.
- **Carrell et al.** *Resilience of clinical text de-identified with "hiding in plain sight" to hostile reidentification attacks
  by human readers.* JAMIA 2020;27(9):1374–1382 (K).
- **BRATsynthetic / Markov-chain replacement** — Osborne et al., arXiv:2210.16125 (2022); journal version *A Markov chain
  replacement strategy for surrogate identifiers: minimizing re-identification risk while preserving text reuse* (PMC12536513,
  2025) (S). Consistent vs. randomized replacement trades text reuse against re-identification risk.
- **Stubbs & Uzuner 2015** (K): the i2b2 2014 surrogate protocol (consistent name replacement, date shifting, realistic IDs).
- **Surrogate substitution preserves PHI detectability: a multi-detector equivalence study.** arXiv:2608.03172 (2026) (V,
  abstract). 1,750 documents, 57k paired spans, 7 languages, 7 benchmarks, 11 detectors; recall on masked spans 76.1% →
  74.9%, **statistically equivalent within ±2 points (TOST, p≈3e-9)**; detector rankings stable; losses come from malformed
  surrogates, not detector degradation. **Directly licenses evaluating detectors on surrogate-substituted / PHI-injected
  corpora**, which is how our primary benchmark is built.
- **On-device PII substitution with small language models** — arXiv:2605.13538 (2026) (S). Locale-conditioned few-shot
  prompting reduces demonstration regurgitation when SLMs generate surrogates.
- **Meystre, Ferrández, Friedlin, South, Shen, Samore.** *Text de-identification for privacy protection: a study of its
  impact on clinical text information content.* JBI 2014;50:142–150 (K). Over-redaction removes clinically relevant content.

**Take-aways.** Surrogation is standard practice for released corpora but LLM-generated surrogates have not been evaluated
for consistency, realism and leakage-masking in combination with a hybrid detector; the equivalence study gives a
methodological template (paired spans, TOST) for our surrogation experiments and for validating the injection benchmark.

---

## 8. Evaluation methodology

- **i2b2 2014 evaluation semantics** (K): strict vs relaxed entity matching; micro P/R/F1 overall and per category; HIPAA-only
  view; token-level variants. We re-implement these semantics in `note_deid.eval`.
- **Privacy-first metrics**: TAB's weighted recall (K); document-level leak rate ("fraction of notes with ≥1 residual
  identifier") used by Philter (K) and Palacios 2026 (V); recall as the primary metric with precision reported for utility.
- **RedactionBench** — arXiv:2606.18782 (2026) (V, abstract). 200 documents / 11 domains / 35 systems (NER models, SLM
  extractors, frontier models with tools). Human agreement: 89.4% on mandatory redactions, 94.1% on safe text, **only 47.7%
  on contextual redactions** — quasi-identifier decisions are inherently ambiguous, which argues for explicit guideline
  policies and for reporting agreement on a "contextual" subset.
- **Information loss / utility**: EMNLP 2025 survey (V) and Meystre 2014 (K); practical proxies are (a) fraction of non-PHI
  clinical tokens redacted, (b) downstream-task drift on de-identified vs original text.
- **Statistics**: paired bootstrap over documents and Holm-corrected multiple comparisons (Dror, Baumer, Shlomov, Reichart,
  *The hitchhiker's guide to testing statistical significance in NLP*, ACL 2018) (K); TOST equivalence testing (V, from
  arXiv:2608.03172); repeated runs to expose LLM non-determinism (Palacios report ±0.001–0.002 F1 over 5 trials) (V).
- **Cost**: tokens and dollars per 1k notes and latency; ensembles cost 3–6× single-pass (V, Palacios).

---

## 9. Datasets and access status (checked 2026-09-14)

| Dataset | Content | Labels | Access | Status / use |
|---|---|---|---|---|
| i2b2/n2c2 2014 de-id (K/V) | 1,304 longitudinal notes, 296 patients | 7 cat / 28 leaf labels | DUA via Harvard DBMI portal, per-user registration | **Portal reported down since 2026-07-28** ("Temporarily Unavailable"; dev.to, Aug 2026, V); i2b2.org datasets page returned HTTP 500 today (V). Primary gated target once access reopens. |
| i2b2 2006 de-id (K) | 889 discharge summaries | 8 classes | same DUA | as above |
| CEGS N-GRID 2016 (K) | 1,000 psychiatric intake records | i2b2-style | same DUA | as above |
| PhysioNet `deid` gold standard (V) | 2,434 MIMIC-II nursing notes with surrogate PHI | names, dates, locations, MRN, phone, company | PhysioNet **credentialed** access (CITI + ID) | Gated; loader stub. |
| MIMIC-IV-Note (K) | ~330k de-identified notes | none (PHI already removed → `___` placeholders) | PhysioNet credentialed | Candidate source for PHI re-injection once credentialed. |
| **MTSamples** (V) | 5,043 transcribed sample reports, 40 specialties; "all names and dates have been changed (or removed)"; placeholders like "Dr. X" | none | Public website; use for educational purposes with attribution to mtsamples.com; Kaggle mirror "Medical Transcriptions" | **Primary source text for the PHI-injection benchmark** after placeholder cleaning; derived text shared only with attribution and for educational/research use. |
| **Asclepius synthetic clinical notes** — Kweon et al., Findings of ACL 2024, arXiv:2309.00237 (K) | ~158k synthetic notes generated from PMC-Patients case reports | none | HF `starmpcc/Asclepius-Synthetic-Clinical-Notes`, CC-BY-NC-SA 4.0 | Secondary source text for injection (discharge-summary style). |
| Synthetic4Health (V) — Frontiers in Digital Health 2025, arXiv:2409.09501, HECTA-UoM | a *generation system* for de-identified synthetic clinical letters, built on MIMIC-IV-Note discharge summaries | clinical entities (SNOMED), not PHI | code on GitHub; underlying letters derive from credentialed MIMIC data | Not an open corpus of notes; not used. |
| **ASQ-PHI** — Data in Brief 2026, PMC12926592 (V) | 1,051 synthetic single-line clinical queries, 2,973 PHI elements, 13 Safe Harbor types | JSON per element | Mendeley Data, **MIT** | Auxiliary (short queries, not notes); useful for over-redaction tests on PHI-free items. |
| SHIELD (V) | 1,381 notes, 10,229 spans, surrogate-replaced | 9 coarse (AGE, DATE, DOCTOR, HOSPITAL, ID, LOCATION, PATIENT, PHONE, WEB) | github.com/susom/shield_dataset | External test set (map 9 categories onto ours) for RQ6. |
| `auren-research/pii-shield` (S) | 531k English documents, 5 domains, silver-labeled by a GLiNER2 PII model; 6 languages | PII (non-clinical) | HF, license TBC | Silver, non-clinical; optional pre-training data only. |
| NVIDIA Nemotron-PII (S) | synthetic PII documents (train set of OpenMed-PII) | ~55 types | HF | Non-clinical PII pre-training / label-mapping reference. |
| AI4Privacy pii-masking-300k / open-pii-masking-500k (K) | multi-domain synthetic PII | ~50 types | HF, open | Non-clinical; optional. |
| MEDDOCAN (K) — Marimon et al., IberLEF 2019 | 1,000 Spanish clinical cases | 29 types | public (CC) | Optional cross-lingual generalization. |
| RedactionBench (V) | 200 documents, 11 domains, contextual redaction | context-dependent | public | Optional contextual-redaction stress test. |

Governance: no dataset files are committed to this repository; gated data stays outside git; the user has confirmed that
commercial LLM APIs may be used for all data in scope of this project (currently only open/synthetic text).

---

## 10. Gap analysis → contributions of this project

1. **Heterogeneous complementarity has not been quantified.** Published ensembles are homogeneous — LLM-only (Palacios 2026:
   correlated errors, no gain) or classical/NER-only (Liu 2017; Kim 2018; Murugadoss 2021; locally augmented ensembles 2024).
   No study measures per-label, per-frequency error overlap between an LLM and a trained token classifier, nor the
   oracle-union headroom.
2. **Label scarcity is under-studied.** Almost all results assume the full i2b2 2014 training set. SHIELD documents the
   rare-category gap of distilled students but does not test hybrids; learning curves vs number of human-labeled notes, and
   recall vs per-label training count, are absent from the de-id literature.
3. **Training-time coupling beyond plain distillation is unexplored for de-id**: rare-label augmentation by controlled PHI
   injection, disagreement-driven active learning, guideline ↔ model co-refinement, and label-set extension of a fixed-taxonomy
   model such as OPF.
4. **Format-induced LLM misses are conflated with semantic misses.** The span-labeling literature shows formats matter; no
   de-id paper separates the two, so "LLMs miss spans" is currently an unexplained aggregate.
5. **No reproducible open benchmark with controllable label frequencies exists**, and the standard gated benchmark is
   currently unreachable. The 2026 surrogate-equivalence result licenses a PHI-injection benchmark built from PHI-free notes.
6. **Surrogation with hybrid detection** has no leakage/consistency/utility evaluation; LLM surrogates are untested at scale.

These six gaps map one-to-one onto the research questions RQ1–RQ8 in `framework-designs.md`.
