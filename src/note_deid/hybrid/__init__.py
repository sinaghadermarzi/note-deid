"""Fusion policies and routing (phase P2): H1 union, H2 label-routed, H3 confidence-gated arbitration,
H4 learned stacking, H5 TC-first LLM editor, H7 selective invocation, H8 audit loop.
Interface: ``Fusion.fuse(doc, outputs: dict[name, list[Span]]) -> list[Span]``.
"""
