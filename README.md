# note-deid

[![ci](https://github.com/sinaghadermarzi/note-deid/actions/workflows/ci.yml/badge.svg)](https://github.com/sinaghadermarzi/note-deid/actions/workflows/ci.yml)
[![code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-DOCS)

Research framework for **hybrid de-identification of clinical notes**: combining a guideline-prompted LLM with a
fine-tuned token classifier (OpenAI Privacy Filter and alternatives) and showing, under a recall-first protocol, when and
why the hybrid beats either system alone — including the rare-label and zero-annotation regimes — with surrogation as a
later arm.

## Status: what exists, what has run

| | State |
|---|---|
| Design | `framework-designs.md`: failure modes, hybrids H1–H16 / S1–S3, research questions RQ1–RQ8, protocol, roadmap, decision log |
| Evaluator | strict / relaxed / typeless matching at subtype, category and binary level; HIPAA view; frequency buckets; paired bootstrap + Holm; complementarity statistics (FN overlap, oracle-union recall, McNemar, kappa). Tested. |
| Fusion | H1 union, H2 label-routed, H3 confidence-gated with verifier / forced-probability hooks, H7 routing features. Tested. |
| Backends | LLM via LiteLLM (formats F1–F4, editor, verifier, parsers, aligners); token classifier via HF Transformers (+ native `opf train` export). Tested offline; the HF model smoke test is opt-in (`NOTE_DEID_HF_TESTS=1`). |
| Benchmark | `synphi` PHI-injection builder with controllable per-label frequencies, open-corpus loaders (MTSamples, Asclepius), demo notes. Tested. |
| Corpus loaders | `i2b2_2014` (XML → unified JSONL; tested on a synthetic fixture, the gated corpus itself has not been loaded). SHIELD, i2b2 2006, N-GRID, PhysioNet: not written. |
| Runner | config-driven `note_deid.run` (detectors → fusion → evaluation → complementarity → `results/`), `note_deid.summarize` for the committed tables |
| **Has run** | only the **offline pipeline check**: `smoke_simulated` on the demo `synphi` build with *simulated* detectors (`results/summary/smoke_simulated*.csv`, see `results/summary/README.md`). It proves the E1 pipeline runs end to end; it is **not** an experimental result. |
| **Has not run** | every paper experiment (E1 complementarity onwards). They need the prior guideline (`prompts/guideline_v0.md`), LLM API keys, a trained token-classifier checkpoint and a GPU node. `experiments/prior/` and `notebooks/` hold only READMEs. |

Next milestone: **E1 figure 1** — build `synphi-v0.1` from MTSamples + Asclepius, fine-tune the token classifier,
run one LLM alias, and publish the FN-overlap table and oracle-union recall that thesis T1 depends on
(`configs/experiments/e1_complementarity.yaml` is the config; the summary goes to `results/summary/e1_complementarity.csv`).

## How the pieces fit

```
 open PHI-free notes ──► cleaning ──► PHI injection (profile: per-label weights, seed) ──► synphi {train,dev,test}.jsonl
 (demo / MTSamples /     strip                 gold spans by construction                      + manifest.json
  Asclepius)             placeholders
 gated corpora (i2b2 2014 XML) ──► note_deid.data.i2b2_2014 convert ──► unified JSONL (stays under data/, never in git)

                       ┌─ TCDetector  (HF token classifier; OPF / DeBERTa / OpenMed, label maps) ─┐
 test split ──────────►┤                                                                          ├──► fusion ──► evaluate
                       └─ LLMDetector (LiteLLM alias + guideline; F1–F4 formats; align to text) ─┘    H1 union        strict / relaxed / typeless
                                                                                                        H2 label-routed  subtype / category / binary
   SimulatedDetector = gold + configured misses / FPs / boundary noise (offline checks)                 H3 conf.-gated   HIPAA view, buckets, bootstrap
                                                                                                                         complementarity across detectors
 results/<exp>/<run-id>/{config.yaml, predictions/, metrics.json, complementarity.json, cost.json, summary.md}
                                     └──► note_deid.summarize ──► results/summary/<exp>.csv (committed)
```

Everything shares one schema (`schema.py`): character offsets with exclusive ends, `Span.source` and `Span.score` so
fusion can arbitrate, and the i2b2-2014 taxonomy (`labels.py`: 7 categories, 28 subtypes, HIPAA subset, OPF / OpenMed
mappings).

## Read first

| Document | Content |
|---|---|
| [`framework-designs.md`](framework-designs.md) | failure-mode analysis, hybrid design space H1–H16 / S1–S3, research questions RQ1–RQ8, evaluation protocol, benchmark plan, backends, roadmap, decision log |
| [`literature.md`](literature.md) | citation-keyed literature review and gap analysis (entries marked (K)/(S) are not yet verified by the author) |
| [`docs/datasets.md`](docs/datasets.md) | dataset access status (2026-09) and the `synphi` PHI-injection benchmark spec |

## Layout

```
src/note_deid/
  schema.py            Doc/Span records (character offsets, exclusive end), JSONL I/O, OPF converters, redaction
  labels.py            i2b2-2014 taxonomy (28 leaf labels), HIPAA subset, OPF-8 / OpenMed mappings, frequency buckets
  eval/                strict/relaxed/typeless matching, entity/token/document metrics, buckets, paired bootstrap,
                       complementarity statistics (RQ1), report writers
  hybrid/              fusion policies H1 union, H2 label-routed, H3 confidence-gated; H7 routing features
  tc/                  span<->tag encoding with chunking, decoding with scores, TCDetector, Trainer script
  llm/                 LiteLLM client (cache, cost log, concurrency 1), formats F1-F4 + editor + verifier, parsers,
                       aligners, LLMDetector
  data/                cleaning, MTSamples/Asclepius loaders, i2b2-2014 XML loader, PHI generators + templates +
                       injector, synphi builder, demo notes
  run.py               config-driven experiment runner (detectors -> fusion -> evaluation -> complementarity -> results/)
  summarize.py         metrics.json / complementarity.json -> the CSV tables committed under results/summary/
configs/               litellm.models.yaml · train/ · synphi/ · experiments/
prompts/               guideline + prompt templates (import the existing guideline as guideline_v0.md)
experiments/prior/     results and notes from the earlier experiments (placeholder)
data/ results/         gitignored working directories (see their READMEs); results/summary/ is committed
scripts/               setup_mac.sh (Apple Silicon / MPS) · setup_a100.sh (4x A100)
tests/                 pytest (offline; model smoke test opt-in with NOTE_DEID_HF_TESTS=1)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm,data]"        # core + LiteLLM + benchmark builder
pip install -e ".[finetune]"            # torch / transformers for the token classifier
pytest -q                               # offline suite
ruff check src tests && ruff format --check src tests
```
Full environment scripts: `scripts/setup_mac.sh`, `scripts/setup_a100.sh` (both also install the `opf` CLI from
github.com/openai/privacy-filter). CI runs the lint, the test suite and the offline quick start below on every push.

## Quick start (offline, no models, no keys)

```bash
# 1. build a small benchmark from the built-in PHI-free demo notes (6 notes x 4 injections = 24 docs)
python -m note_deid.data.synphi build --profile configs/synphi/i2b2like.yaml --out data/synphi/demo --demo --repeat 4
# 2. run the pipeline with simulated detectors: fusion, metrics, bootstrap, complementarity
python -m note_deid.run --config configs/experiments/smoke_simulated.yaml
# results/smoke_simulated/<run-id>/{summary.md, metrics.json, complementarity.json, predictions/}
# 3. flatten a run into the summary tables
python -m note_deid.summarize results/smoke_simulated/<run-id> --out results/summary/smoke_simulated
```

## Real runs (need data, keys, hardware)

```bash
# benchmark from open notes (mtsamples.csv from the public mirror; Asclepius streamed from the HF hub)
python -m note_deid.data.synphi build --profile configs/synphi/i2b2like.yaml --out data/synphi/v0.1 \
    --mtsamples data/raw/mtsamples/mtsamples.csv --asclepius 1000 --seed 0
# gated i2b2-2014 records (under the DUA, on the machine they were downloaded to) -> unified JSONL
python -m note_deid.data.i2b2_2014 convert data/raw/i2b2_2014/testing-PHI-Gold-fixed --out data/i2b2_2014/test.jsonl
# token classifier, HF path: 28-label head fine-tuned on the unified train/dev JSONL (OPF or a DeBERTa base model)
python -m note_deid.tc.train --config configs/train/opf_finetune.yaml
# token classifier, native-OPF path: export OPF-shaped JSONL (8 labels; unsupported subtypes dropped), then the
# upstream CLI. Such a checkpoint needs `label_map: opf_category` and is only comparable at `levels: [category]`.
python -m note_deid.data.synphi export-opf --bench data/synphi/v0.1
opf train data/synphi/v0.1/train.opf.jsonl --output-dir models/opf-native-synphi-v0.1
# E1 complementarity (LLM alias from configs/litellm.models.yaml; keys from the environment)
python -m note_deid.run --config configs/experiments/e1_complementarity.yaml
```

## Backends

- **LLM**: [LiteLLM](https://github.com/BerriAI/litellm) with model aliases in `configs/litellm.models.yaml`
  (commercial APIs; optional OpenAI-compatible local server). Keys via environment variables only. Disk cache on;
  request concurrency defaults to 1; provider-unsupported parameters are dropped.
- **Token classification**: Hugging Face Transformers (`AutoModelForTokenClassification`) for OPF, DeBERTa-v3 and
  OpenMed-PII; OPF also via `opf train`. Apple Silicon (MPS) for development-scale runs, 4x A100 80GB for full runs.

## Data governance

No dataset files are committed. Gated corpora (i2b2/n2c2, PhysioNet) stay outside git and are used only under their
agreements; the `i2b2_2014` converter writes under the gitignored `data/`. The primary benchmark (`synphi`) is built
from open, PHI-free notes with injected synthetic PHI.

## Roadmap

P0 design → **P1 (now)** benchmark + evaluator + both baselines + complementarity analysis → P2 inference-time hybrids →
P3 training-time hybrids and learning curves → P4 surrogation → P5 robustness and paper. Details in
`framework-designs.md` §8. Progress is tracked in the status table above and in the decision log (§10).

## License

- **Code**: Apache License 2.0 (`LICENSE`, `NOTICE`).
- **Documents, prompts, guidelines and fully synthetic data** (the built-in demo notes and any `synphi` build made only
  from them): Creative Commons Attribution 4.0 International (`LICENSE-DOCS`).
- **Third-party corpora keep their own terms.** A `synphi` build that contains Asclepius notes inherits CC BY-NC-SA 4.0,
  and MTSamples-derived notes are for educational use with attribution to mtsamples.com, so those builds are shared as a
  recipe (profile, seed, `manifest.json`) rather than as text. Gated corpora are never redistributed. Details in
  `docs/datasets.md` §4.

Cite the software with `CITATION.cff`.
