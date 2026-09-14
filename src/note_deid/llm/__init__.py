"""LiteLLM-backed detector (phase P1).

Planned modules:
- ``client``      completion wrapper: model aliases from configs/litellm.models.yaml, disk cache, retries,
                  cost logging, concurrency (default 1)
- ``prompts``     guideline + task templates; output formats F1 (JSON strings), F2 (anchored JSON),
                  F3 (inline tags), F4 (segmented inline tags); editor (H5) and verifier (H3) prompts
- ``parse``       tolerant JSON / tag parsing
- ``align``       exact, anchored and fuzzy alignment of LLM output to character offsets; diff validation for F3/F4
- ``detector``    ``LLMDetector.detect(docs) -> list[list[Span]]`` with grounding scores
"""
