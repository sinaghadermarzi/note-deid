# note-deid

Research framework for **hybrid de-identification of clinical notes**: combining a guideline-prompted LLM with a
fine-tuned token classifier (OpenAI Privacy Filter and alternatives) and showing, under a recall-first protocol, when and
why the hybrid beats either system alone — including the rare-label and zero-annotation regimes — with surrogation as a
later arm.

Status: **phase P1 infrastructure** (evaluator, fusion, both backends, benchmark builder, runner) is in place and
tested offline. No paper experiments have been run yet: they need the prior guideline, API keys, and a GPU node.

## Read first

| Document | Content |
|---|---|
| [`framework-designs.md`](framework-designs.md) | failure-mode analysis, hybrid design space H1–H16 / S1–S3, research questions RQ1–RQ8, evaluation protocol, benchmark plan, backends, roadmap, decision log |
| [`literature.md`](literature.md) | citation-keyed literature review and gap analysis |
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
  data/                cleaning, MTSamples/Asclepius loaders, PHI generators + templates + injector, synphi builder,
                       demo notes
  run.py               config-driven experiment runner (detectors -> fusion -> evaluation -> complementarity -> results/)
configs/               litellm.models.yaml · train/ · synphi/ · experiments/
prompts/               guideline + prompt templates (import the existing guideline as guideline_v0.md)
experiments/prior/     results and notes from the earlier experiments
data/ results/         gitignored working directories (see their READMEs)
scripts/               setup_mac.sh (Apple Silicon / MPS) · setup_a100.sh (4x A100)
tests/                 pytest (offline; model smoke test opt-in with NOTE_DEID_HF_TESTS=1)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm,data]"        # core + LiteLLM + benchmark builder
pip install -e ".[finetune]"            # torch / transformers for the token classifier
pytest -q
```
Full environment scripts: `scripts/setup_mac.sh`, `scripts/setup_a100.sh` (both also install the `opf` CLI from
github.com/openai/privacy-filter).

## Quick start (offline)

```bash
# 1. build a tiny benchmark from the built-in PHI-free demo notes
python -m note_deid.data.synphi build --profile configs/synphi/i2b2like.yaml --out data/synphi/demo --demo
# 2. run the pipeline with simulated detectors: fusion, metrics, bootstrap, complementarity
python -m note_deid.run --config configs/experiments/smoke_simulated.yaml
# results/smoke_simulated/<run-id>/{summary.md, metrics.json, complementarity.json, predictions/}
```

## Real runs (need data, keys, hardware)

```bash
# benchmark from open notes (mtsamples.csv from the public mirror; Asclepius streamed from the HF hub)
python -m note_deid.data.synphi build --profile configs/synphi/i2b2like.yaml --out data/synphi/v0.1 \
    --mtsamples data/raw/mtsamples/mtsamples.csv --asclepius 1000 --seed 0
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
agreements. The primary benchmark (`synphi`) is built from open, PHI-free notes with injected synthetic PHI.

## Roadmap

P0 design → **P1 (now)** benchmark + evaluator + both baselines + complementarity analysis → P2 inference-time hybrids →
P3 training-time hybrids and learning curves → P4 surrogation → P5 robustness and paper. Details in
`framework-designs.md` §8.

## License

- **Code**: Apache License 2.0 (`LICENSE`, `NOTICE`).
- **Documents, prompts, guidelines and fully synthetic data** (the built-in demo notes and any `synphi` build made only
  from them): Creative Commons Attribution 4.0 International (`LICENSE-DOCS`).
- **Third-party corpora keep their own terms.** A `synphi` build that contains Asclepius notes inherits CC BY-NC-SA 4.0,
  and MTSamples-derived notes are for educational use with attribution to mtsamples.com, so those builds are shared as a
  recipe (profile, seed, `manifest.json`) rather than as text. Gated corpora are never redistributed. Details in
  `docs/datasets.md` §4.

Cite the software with `CITATION.cff`.
