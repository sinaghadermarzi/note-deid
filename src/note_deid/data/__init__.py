"""Corpus loaders and the ``synphi`` PHI-injection benchmark.

- ``samples``            built-in PHI-free demo notes (offline tests and smoke runs)
- ``mtsamples``, ``asclepius``   open PHI-free source notes
- ``cleaning``           placeholder stripping and rule-labeled residual quasi-identifiers
- ``phi_injection``      generators, templates, injector (gold by construction)
- ``synphi``             build CLI: profiles, stratified splits, manifest
Gated corpora (i2b2 2014/2006, N-GRID 2016, PhysioNet deid) get loaders once access exists (docs/datasets.md).
"""

from note_deid.data.cleaning import residual_spans, strip_placeholders
from note_deid.data.phi_injection.inject import Injector, Profile, inject_all
from note_deid.data.samples import demo_docs
from note_deid.data.synphi import build, stratified_split

__all__ = [
    "Injector",
    "Profile",
    "build",
    "demo_docs",
    "inject_all",
    "residual_spans",
    "strip_placeholders",
    "stratified_split",
]
