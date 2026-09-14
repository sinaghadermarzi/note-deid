"""note_deid: hybrid LLM + token-classifier de-identification of clinical notes.

Package layout (see framework-designs.md §7):

- ``schema``    unified Doc/Span records, JSONL I/O, converters (OpenAI Privacy Filter JSONL)
- ``labels``    i2b2-2014 taxonomy, HIPAA subset, mapping tables to OPF-8 and OpenMed-PII
- ``data``      corpus loaders and the ``synphi`` PHI-injection benchmark builder      (phase P1)
- ``llm``       LiteLLM-backed detector: prompts, output formats F1-F4, aligners        (phase P1)
- ``tc``        token-classification detector: OPF / DeBERTa training and inference    (phase P1)
- ``hybrid``    fusion policies H1-H8 and routing                                       (phase P2)
- ``surrogate`` surrogate generation S1-S3                                              (phase P4)
- ``eval``      matching semantics, metrics, bootstrap, error taxonomy, cost reports     (phase P1)
"""

from note_deid.schema import Doc, Span

__all__ = ["Doc", "Span"]
__version__ = "0.0.1"
