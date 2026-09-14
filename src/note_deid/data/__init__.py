"""Corpus loaders and the ``synphi`` PHI-injection benchmark builder (phase P1).

Planned modules (docs/datasets.md):
- ``mtsamples``, ``asclepius``          open PHI-free source notes
- ``phi_injection.generators``          per-subtype entity generators (Faker, Synthea bundles, curated lists)
- ``phi_injection.insert``              slot templates, in-text replacement, LLM contextual insertion
- ``synphi``                            profiles, splits, manifest, versioning
- ``i2b2_2014``, ``i2b2_2006``, ``ngrid_2016``, ``physionet_deid``   gated corpora (stubs until access)
- ``asq_phi``                           auxiliary over-redaction set
All loaders yield ``note_deid.schema.Doc`` with canonical labels (``note_deid.labels``).
"""
