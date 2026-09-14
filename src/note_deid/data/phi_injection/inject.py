"""The injector: header/signature slots, in-text replacement of generic mentions, sentence insertions at
paragraph ends, controlled per-subtype counts from a frequency profile, gold spans by construction."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from note_deid.data.cleaning import residual_spans, strip_placeholders
from note_deid.data.phi_injection.generators import Generators, PatientBundle
from note_deid.data.phi_injection.templates import HEADERS, SENTENCES, SIGNATURES, render
from note_deid.schema import Doc, Span

_GENERIC = {
    "PATIENT": re.compile(r"\b[Tt]he patient\b"),
    "HOSPITAL": re.compile(r"\b[Tt]he hospital\b"),
    "DOCTOR": re.compile(r"\b[Tt]he (?:physician|doctor)\b"),
}


@dataclass
class Profile:
    version: str = "0.1"
    target_density: float = 1.0
    institutional_group: bool = False
    weights: dict[str, float] = field(default_factory=dict)
    p_header: float = 0.8
    p_signature: float = 0.7
    replace_rates: dict[str, float] = field(default_factory=lambda: {"PATIENT": 0.35, "HOSPITAL": 0.5, "DOCTOR": 0.3})
    keep_residual: bool = True

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        with Path(path).open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(
            version=str(raw.get("version", "0.1")),
            target_density=float(raw.get("target_density", 1.0)),
            institutional_group=bool(raw.get("institutional_group", False)),
            weights={str(k): float(v) for k, v in (raw.get("weights") or {}).items()},
            p_header=float(raw.get("p_header", 0.8)),
            p_signature=float(raw.get("p_signature", 0.7)),
            replace_rates={str(k): float(v) for k, v in (raw.get("replace_rates") or {}).items()}
            or {"PATIENT": 0.35, "HOSPITAL": 0.5, "DOCTOR": 0.3},
            keep_residual=bool(raw.get("keep_residual", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target_density": self.target_density,
            "institutional_group": self.institutional_group,
            "weights": dict(self.weights),
            "p_header": self.p_header,
            "p_signature": self.p_signature,
            "replace_rates": dict(self.replace_rates),
            "keep_residual": self.keep_residual,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(yaml.safe_dump(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:  # normal approximation is fine for our densities
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class _Builder:
    """Accumulates text pieces and their spans, tracking absolute offsets."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[Span] = []
        self.length = 0

    def add(self, text: str, spans: Sequence[Span] = ()) -> None:
        for s in spans:
            self.spans.append(Span(self.length + s.start, self.length + s.end, s.label, s.text, s.source, s.score))
        self.parts.append(text)
        self.length += len(text)

    def text(self) -> str:
        return "".join(self.parts)


def _word_present(text: str, s: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])", text) is not None


class Injector:
    def __init__(self, profile: Profile, seed: int = 0) -> None:
        self.profile = profile
        self.seed = seed
        self.gens = Generators(seed)

    def _bundle_for(self, text: str, rng: random.Random) -> PatientBundle:
        """Draw a bundle whose name/organization strings do not already occur in the source text."""
        for _ in range(20):
            b = self.gens.bundle()
            strings = [s for vals in b.strings().values() for s in vals]
            if not any(_word_present(text, s) for s in strings):
                return b
        return b  # give up after 20 draws; collisions are recorded in meta

    def _targets(self, n_tokens: int, rng: random.Random) -> dict[str, int]:
        scale = self.profile.target_density * n_tokens / 1000.0
        targets: dict[str, int] = {}
        for label, w in self.profile.weights.items():
            if label == "INSTITUTIONAL" and not self.profile.institutional_group:
                continue
            targets[label] = poisson(rng, w * scale)
        return targets

    def inject(self, doc: Doc) -> Doc:
        rng = random.Random(f"{self.seed}:{doc.doc_id}")
        self.gens.rng = rng  # per-document determinism, independent of call order
        self.gens.fake.seed_instance(f"{self.seed}:{doc.doc_id}")
        source_text, n_placeholders = strip_placeholders(doc.text)
        bundle = self._bundle_for(source_text, rng)
        fill = lambda label, form: self.gens.mention(label, bundle, form)  # noqa: E731
        builder = _Builder()
        placed: dict[str, int] = {}

        def count(spans: Sequence[Span]) -> None:
            for s in spans:
                placed[s.label] = placed.get(s.label, 0) + 1

        # 1. header
        if rng.random() < self.profile.p_header:
            text, spans = render(rng.choice(HEADERS), fill)
            count(spans)
            builder.add(text + "\n\n", spans)

        # 2. body: residual quasi-identifiers, generic-mention replacement, sentence insertions per paragraph
        targets = self._targets(len(source_text.split()), rng)
        paragraphs = re.split(r"(\n\s*\n)", source_text)  # keep separators
        residual = residual_spans(source_text) if self.profile.keep_residual else []
        res_by_offset = {(s.start, s.end): s for s in residual}
        para_offset = 0
        body_paras: list[tuple[str, list[Span]]] = []
        for piece in paragraphs:
            if re.fullmatch(r"\n\s*\n", piece):
                body_paras.append((piece, []))
                para_offset += len(piece)
                continue
            p_spans = [
                Span(s.start - para_offset, s.end - para_offset, s.label, s.text, s.source)
                for (a, b), s in res_by_offset.items()
                if a >= para_offset and b <= para_offset + len(piece)
            ]
            new_text, new_spans = self._replace_generic(piece, p_spans, fill, rng)
            body_paras.append((new_text, new_spans))
            para_offset += len(piece)
        for _, spans in body_paras:
            count(spans)

        # sentence insertions to reach the sampled targets, appended to random paragraphs
        # insertion targets: real paragraphs only (skip separators and short header-like lines)
        text_paras = [
            i for i, (t, _) in enumerate(body_paras) if not re.fullmatch(r"\n\s*\n", t) and len(t.split()) >= 12
        ]
        if not text_paras:
            text_paras = [i for i, (t, _) in enumerate(body_paras) if not re.fullmatch(r"\n\s*\n", t) and t.strip()]
        for label, target in targets.items():
            bank = SENTENCES.get(label)
            if not bank or not text_paras:
                continue
            for _ in range(max(0, target - placed.get(label, 0))):
                sent, spans = render(rng.choice(bank), fill)
                i = rng.choice(text_paras)
                base_text, base_spans = body_paras[i]
                sep = "" if base_text.endswith((" ", "\n")) else " "
                offset = len(base_text) + len(sep)
                shifted = [Span(offset + s.start, offset + s.end, s.label, s.text, s.source) for s in spans]
                body_paras[i] = (base_text + sep + sent, base_spans + shifted)
                count(spans)
        for text, spans in body_paras:
            builder.add(text, spans)

        # 3. signature block
        if rng.random() < self.profile.p_signature:
            text, spans = render(rng.choice(SIGNATURES), fill)
            count(spans)
            builder.add("\n\n")
            builder.add(text, spans)

        final = builder.text()
        spans = sorted(builder.spans, key=lambda s: (s.start, s.end))
        meta = dict(doc.meta)
        meta.update(
            {
                "injected": True,
                "profile": self.profile.version,
                "profile_fingerprint": self.profile.fingerprint(),
                "n_placeholders_stripped": n_placeholders,
                "n_residual_rule_spans": len(residual),
                "patient": bundle.patient.full,
                "targets": targets,
                "placed": placed,
            }
        )
        out = Doc(doc.doc_id, final, spans, meta).validate()
        _check_unlabeled_occurrences(out, bundle)
        return out

    def _replace_generic(self, text: str, existing: list[Span], fill, rng: random.Random) -> tuple[str, list[Span]]:
        """Replace a fraction of generic mentions ("the patient", "the hospital", "the physician") with PHI."""
        edits: list[tuple[int, int, str, Span | None]] = []
        for label, pat in _GENERIC.items():
            rate = self.profile.replace_rates.get(label, 0.0)
            for m in pat.finditer(text):
                if rng.random() >= rate:
                    continue
                if any(s.start < m.end() and m.start() < s.end for s in existing):
                    continue
                surface, labeled = fill(label, "full" if label != "PATIENT" else rng.choice(["full", None, None]))
                if m.group()[0] == "T" and surface[0].islower():
                    surface = surface[0].upper() + surface[1:]
                off = surface.index(labeled)
                edits.append((m.start(), m.end(), surface, Span(off, off + len(labeled), label, labeled, "gold")))
        edits.sort()
        # drop overlapping edits
        kept: list[tuple[int, int, str, Span | None]] = []
        last_end = -1
        for e in edits:
            if e[0] >= last_end:
                kept.append(e)
                last_end = e[1]
        out: list[str] = []
        spans: list[Span] = []
        pos = 0
        length = 0
        old_spans = sorted(existing, key=lambda s: s.start)

        def shift_existing(upto: int, delta: int) -> None:
            pass  # existing spans are re-based below

        for start, end, surface, span in kept:
            chunk = text[pos:start]
            out.append(chunk)
            length += len(chunk)
            if span is not None:
                spans.append(Span(length + span.start, length + span.end, span.label, span.text, span.source))
            out.append(surface)
            length += len(surface)
            pos = end
        out.append(text[pos:])
        new_text = "".join(out)
        # re-base the pre-existing (residual) spans through the edits
        for s in old_spans:
            delta = 0
            for start, end, surface, _ in kept:
                if end <= s.start:
                    delta += len(surface) - (end - start)
            spans.append(Span(s.start + delta, s.end + delta, s.label, s.text, s.source))
        return new_text, sorted(spans, key=lambda s: (s.start, s.end))


def _check_unlabeled_occurrences(doc: Doc, bundle: PatientBundle) -> None:
    """Record (in meta) injected strings that also occur unlabeled, so ambiguous gold can be audited."""
    labeled = {(s.start, s.end) for s in doc.spans}
    ambiguous: list[str] = []
    for values in bundle.strings().values():
        for v in values:
            for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", doc.text):
                covered = any(a <= m.start() and m.end() <= b for a, b in labeled)
                if not covered:
                    ambiguous.append(v)
                    break
    doc.meta["ambiguous_strings"] = sorted(set(ambiguous))


def inject_all(docs: Sequence[Doc], profile: Profile, seed: int = 0) -> list[Doc]:
    inj = Injector(profile, seed)
    return [inj.inject(d) for d in docs]


__all__ = ["Injector", "Profile", "inject_all", "poisson", "Mapping"]
