"""Token-classification detector over Hugging Face ``AutoModelForTokenClassification``.

Works for OpenAI Privacy Filter (native 128k context), DeBERTa-v3 / BERT-family models (windowed with stride) and
OpenMed-PII. Model labels are mapped to the canonical taxonomy through ``label_map`` (e.g. ``OPF_TO_I2B2_CATEGORY``
or ``OPENMED_TO_I2B2``); a label mapping to ``None`` is dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from note_deid.labels import map_label, strip_bio_prefix
from note_deid.schema import Doc, Span
from note_deid.tc.decode import decode_tags


@dataclass
class TokenPrediction:
    offsets: list[tuple[int, int]]
    tags: list[str]
    probs: list[float]  # probability of the predicted tag per token


class TCDetector:
    name = "tc"

    def __init__(
        self,
        model_name_or_path: str,
        label_map: Mapping[str, str | None] | None = None,
        device: str | None = None,
        max_length: int | None = None,
        stride: int = 64,
        batch_size: int = 8,
        score: str = "min",
        tokenizer: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or model_name_or_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        limit = getattr(self.tokenizer, "model_max_length", None) or 512
        self.max_length = min(max_length or limit, limit)
        self.stride = stride
        self.batch_size = batch_size
        self.score = score
        self.label_map = label_map

    def _canonical(self, span: Span) -> Span | None:
        if self.label_map is None:
            return span
        mapped = map_label(span.label, self.label_map)
        if mapped is None:
            return None
        return Span(span.start, span.end, mapped, span.text, span.source, span.score)

    def predict_tokens(self, text: str) -> TokenPrediction:
        import torch

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        offsets_all = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping", None)
        n_windows = enc["input_ids"].shape[0]
        label_ids: list[list[int]] = []
        probs: list[list[float]] = []
        with torch.no_grad():
            for b in range(0, n_windows, self.batch_size):
                batch = {k: v[b : b + self.batch_size].to(self.device) for k, v in enc.items()}
                logits = self.model(**batch).logits
                p = torch.softmax(logits.float(), dim=-1)
                conf, ids = p.max(dim=-1)
                label_ids.extend(ids.cpu().tolist())
                probs.extend(conf.cpu().tolist())
        # Flatten windows into one token stream keyed by character offsets (drop padding / special tokens).
        tokens: dict[tuple[int, int], list[tuple[int, int, float]]] = {}  # offset -> [(centrality, id, prob)]
        for w in range(n_windows):
            mask = enc["attention_mask"][w].tolist()
            real = [i for i, (o, m) in enumerate(zip(offsets_all[w], mask, strict=True)) if m and o[0] != o[1]]
            for k, i in enumerate(real):
                centrality = min(k, len(real) - 1 - k)
                tokens.setdefault(tuple(offsets_all[w][i]), []).append((centrality, label_ids[w][i], probs[w][i]))
        ordered = sorted(tokens)
        best = [max(tokens[o]) for o in ordered]
        return TokenPrediction(
            offsets=[tuple(o) for o in ordered],
            tags=[self.id2label[i] for _, i, _ in best],
            probs=[p for _, _, p in best],
        )

    def token_max_probs(self, text: str) -> list[float]:
        """Per-token confidence of the predicted class (for H7 routing features)."""
        return self.predict_tokens(text).probs

    def detect_one(self, doc: Doc) -> list[Span]:
        tp = self.predict_tokens(doc.text)
        raw = decode_tags(tp.tags, tp.offsets, tp.probs, doc.text, source=self.name, score=self.score)
        out = []
        for s in raw:
            c = self._canonical(s)
            if c is not None:
                out.append(c)
        return out

    def detect(self, docs: Iterable[Doc]) -> list[list[Span]]:
        return [self.detect_one(d) for d in docs]

    def model_labels(self) -> list[str]:
        return sorted({strip_bio_prefix(v) for v in self.id2label.values() if v != "O"})


__all__ = ["TCDetector", "TokenPrediction"]
