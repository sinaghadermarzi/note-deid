# Datasets: access status, licenses, and the `synphi` benchmark

Last checked: 2026-09-14. See `literature.md` §9 for citations and `framework-designs.md` §6 for the rationale.

## 1. Access status

| Dataset | Kind | Size | Labels | License / access | Status (2026-09-14) | Planned loader |
|---|---|---|---|---|---|---|
| i2b2/n2c2 2014 de-id | real, gated | 1,304 notes / 296 patients | 7 cat / 28 leaf labels (XML offsets) | DUA via Harvard DBMI Data Portal, per-user | **Portal "Temporarily Unavailable" since ~2026-07-28; i2b2.org HTTP 500** | `note_deid.data.i2b2_2014` (stub) |
| i2b2 2006 de-id | real, gated | 889 discharge summaries | 8 classes | same DUA | as above | `note_deid.data.i2b2_2006` (stub) |
| CEGS N-GRID 2016 | real, gated | 1,000 psychiatric intake records | i2b2-style | same DUA | as above | `note_deid.data.ngrid_2016` (stub) |
| PhysioNet `deid` gold standard v1.1 | real, gated | 2,434 MIMIC-II nursing notes (surrogate PHI) | names, dates, locations, MRN, phone, company | PhysioNet credentialed access (CITI + identity check); ODC-By 1.0 for files | available to credentialed users | `note_deid.data.physionet_deid` (stub) |
| MIMIC-IV-Note v2.2 | real, gated, already de-identified | ~330k notes | none (`___` placeholders) | PhysioNet credentialed | available to credentialed users | re-injection source (later) |
| **MTSamples** | real-style transcribed samples, names/dates changed | 5,043 reports, 40 specialties | none | public website; educational use with attribution to mtsamples.com; Kaggle mirror "Medical Transcriptions" | available | `note_deid.data.mtsamples` |
| **Asclepius synthetic clinical notes** | synthetic notes from PMC-Patients | ~158k | none | HF `starmpcc/Asclepius-Synthetic-Clinical-Notes`, CC-BY-NC-SA 4.0 | available | `note_deid.data.asclepius` |
| Synthetic4Health | generation system over MIMIC-IV-derived letters | 204 letters | clinical entities | code public; source data credentialed | not an open note corpus | not used |
| ASQ-PHI | synthetic single-line clinical queries | 1,051 queries / 2,973 PHI elements / 13 types | JSON per element | Mendeley Data, MIT | available | `note_deid.data.asq_phi` (auxiliary) |
| SHIELD | surrogate-replaced notes, LLM pre-annotation + human adjudication (Stanford Medicine) | 1,381 notes / 10,229 spans | 9 coarse (AGE, DATE, DOCTOR, HOSPITAL, ID, LOCATION, PATIENT, PHONE, WEB) | github.com/susom/shield_dataset | available | `note_deid.data.shield` (P2; external test, RQ6) |
| `auren-research/pii-shield` | real docs, silver PII labels, 5 domains, 6 languages | 531k English docs | PII (non-clinical) | HF; license TBC | available | optional pre-training only |
| NVIDIA Nemotron-PII | synthetic PII documents | 50k/5k/45k | ~55 types | HF | available | label-map reference / optional pre-training |
| AI4Privacy pii-masking-300k / open-pii-masking-500k | synthetic multi-domain PII | 300k–500k | ~50 types | HF, open | available | optional |
| MEDDOCAN | Spanish clinical cases | 1,000 | 29 types | public | available | optional cross-lingual |
| RedactionBench | multi-domain contextual redaction | 200 docs / 11 domains | contextual | public | available | optional stress test |

Rules: no dataset files are committed; everything lives under `data/` (gitignored) with a `data/README.md` describing the
expected layout; gated data never leaves the machine it was downloaded to except as allowed by its DUA.

## 2. Canonical label set

The canonical labels are the i2b2-2014 subtypes (see `src/note_deid/labels.py`). Every loader maps its native labels to
this set; `labels.py` also maps to the OpenAI Privacy Filter 8-label space (for `opf train`) and to OpenMed-PII.

## 3. `synphi` — PHI-injection benchmark (v0 spec)

Goal: a shareable, reproducible clinical de-identification benchmark with **controllable per-label frequencies**, built by
injecting synthetic PHI into PHI-free notes. Validity argument: detector recall on surrogate-substituted text is
statistically equivalent to original text (arXiv 2608.03172); realism is additionally checked by human spot-check and a
discriminator test; real-data transfer is evaluated when gated corpora become available.

### 3.1 Pipeline

1. **Ingest** MTSamples and Asclepius; keep `source`, `note_type`, `specialty` metadata.
2. **Clean**: remove placeholders ("Dr. X", "(Patient name)", template DOBs, bracketed tokens); flag residual PHI-like
   strings with OPF + regex + one LLM pass; drop unresolved notes.
3. **Generate entities** with `note_deid.data.phi_injection.generators`:
   - Faker: names (with international / uncommon name lists), streets, cities, states, zips, countries, phones, fax,
     emails, URLs, IPs, SSN-shaped, license and VIN shapes.
   - Synthea bundles (optional): coherent patient (name, DOB, address, MRN, providers, organizations).
   - Curated lists: hospitals, organizations, professions (O*NET titles), institution lexicon (department/clinic codes,
     building names, local abbreviations) for the optional `INSTITUTIONAL` group.
   - Ages including ≥ 90; MRN / health-plan / account / device / bio-id shapes.
4. **Insert** via slot templates (identification blocks, letterheads, signature/"Dictated by"/"cc:" lines), in-text
   replacement of generic mentions, and LLM contextual insertion (sentence rewrite returning tagged text; entity must appear
   verbatim; any other change is rejected). Enforce document-level consistency (repeated mentions with variants; coherent
   dates; DOB/age agreement).
5. **Gold**: offsets by construction; every occurrence of an injected string is labeled; assertions in the builder.
6. **Profiles** (`configs/synphi/*.yaml`): `i2b2like`, `uniform`, `tailstress` (evaluation only), `decoupled`
   (train profile ≠ test profile).
7. **Split** by source document, stratified by note type; fixed seeds; write `manifest.json` (source ids, generator
   config, seed, profile, version) so the corpus regenerates exactly.
8. **Validate**: human spot-check (100 notes, naturalness 1–5); discriminator test on inserted vs original sentences;
   injection-density report per label.

### 3.2 Output format

JSONL of `Doc` records (`src/note_deid/schema.py`): `{"doc_id", "text", "spans": [{"start", "end", "label", "text"}],
"meta": {"source", "note_type", "profile", "injected": true}}`. Converters produce OPF training JSONL and BIO/BIOES token
files for HF training.

### 3.3 Versioning

`synphi-v0.1` = first build used in E1/E8; any change to sources, cleaning, generators or profiles bumps the version.
Results always cite the version and manifest hash.
