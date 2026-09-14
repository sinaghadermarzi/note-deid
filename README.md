# note-deid

Research framework for **hybrid de-identification of clinical notes**: combining a guideline-prompted LLM with a
fine-tuned token classifier (OpenAI Privacy Filter and alternatives) and showing, under a recall-first protocol, when and
why the hybrid beats either system alone — including the rare-label and zero-annotation regimes — with surrogation as a
later arm.

Status: **design phase (P0)**. No experiments run yet.

## Read first

| Document | Content |
|---|---|
| [`framework-designs.md`](framework-designs.md) | failure-mode analysis, hybrid design space H1–H16 / S1–S3, research questions RQ1–RQ8, evaluation protocol, benchmark plan, backends, roadmap, decision log |
| [`literature.md`](literature.md) | citation-keyed literature review and gap analysis |
| [`docs/datasets.md`](docs/datasets.md) | dataset access status (2026-09) and the `synphi` PHI-injection benchmark spec |

## Layout

```
src/note_deid/        schema.py (Doc/Span, JSONL, OPF converters) · labels.py (taxonomy + mappings)
                      data/ llm/ tc/ hybrid/ surrogate/ eval/   (module docstrings describe phase-P1+ contents)
configs/              litellm.models.yaml · train/ · synphi/ · experiments/
prompts/              guideline + prompt templates (import the existing guideline as guideline_v0.md)
experiments/prior/    results and notes from the earlier experiments
data/ results/        gitignored working directories (see their READMEs)
scripts/              setup_mac.sh (Apple Silicon / MPS) · setup_a100.sh (4x A100)
tests/                pytest
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # schema + labels + tests
pip install -e ".[llm,finetune,data]"   # backends (torch/transformers/litellm/faker)
pytest -q
```
Full environment scripts: `scripts/setup_mac.sh`, `scripts/setup_a100.sh` (both also install the `opf` CLI from
github.com/openai/privacy-filter).

## Backends

- **LLM**: [LiteLLM](https://github.com/BerriAI/litellm) with model aliases in `configs/litellm.models.yaml`
  (commercial APIs; optional OpenAI-compatible local server). Keys via environment variables only. Request concurrency
  defaults to 1.
- **Token classification**: Hugging Face Transformers (`AutoModelForTokenClassification`) for OPF, DeBERTa-v3 and
  OpenMed-PII; OPF also via `opf train`. Apple Silicon (MPS) for development-scale runs, 4x A100 80GB for full runs.

## Data governance

No dataset files are committed. Gated corpora (i2b2/n2c2, PhysioNet) stay outside git and are used only under their
agreements. The primary benchmark (`synphi`) is built from open, PHI-free notes with injected synthetic PHI.

## Roadmap

P0 design (this) → P1 benchmark + evaluator + both baselines + complementarity analysis → P2 inference-time hybrids →
P3 training-time hybrids and learning curves → P4 surrogation → P5 robustness and paper. Details in
`framework-designs.md` §8.
