"""Token-classification detector (phase P1).

Planned modules:
- ``encode``      Doc spans -> token labels (BIO/BIOES) with offset mapping; window/stride chunking for 512-token
                  encoders (OPF needs no chunking: 128k context)
- ``decode``      token probabilities -> spans with scores (min/mean token probability)
- ``train``       HF Trainer path (DeBERTa-v3, OpenMed-PII, OPF with extended head for H13) and the
                  ``opf train`` CLI path via ``schema.to_opf_record``; configs under configs/train/
- ``detector``    ``TCDetector.detect(docs) -> list[list[Span]]``
"""
