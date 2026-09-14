"""Fine-tune a token classifier with the Hugging Face ``Trainer``.

    python -m note_deid.tc.train --config configs/train/opf_finetune.yaml [--seed 13] [--max-steps 50]

Config keys (YAML): base_model, tokenizer (optional override, defaults to base_model), label_space (i2b2_28 |
i2b2_categories | opf8 | binary), train_jsonl, dev_jsonl,
output_dir, learning_rate, epochs, per_device_batch_size, gradient_accumulation, max_length, stride, precision
(bf16 | fp16 | fp32), seeds (list; the first is used unless --seed is given), scheme (bioes | bio).

``train_jsonl`` / ``dev_jsonl`` are the unified files written by ``synphi build``; gold labels are mapped into the
chosen label space here. For the native OpenAI Privacy Filter label space the alternative path is the upstream CLI on
the export written by ``python -m note_deid.data.synphi export-opf``: ``opf train <train.opf.jsonl>``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from note_deid.labels import (
    I2B2_2014_CATEGORIES,
    I2B2_2014_SUBTYPES,
    I2B2_TO_OPF,
    OPF_LABELS,
    SUBTYPE_TO_CATEGORY,
    bioes_tags,
)
from note_deid.schema import Doc, Span, read_jsonl
from note_deid.tc.encode import IGNORE_INDEX, chunk_windows, spans_to_token_labels, tags_to_ids


def label_space(name: str) -> tuple[list[str], Mapping[str, str | None] | None]:
    """(entity labels, mapping applied to gold spans before tagging)."""
    if name == "i2b2_28":
        return list(I2B2_2014_SUBTYPES), None
    if name == "i2b2_categories":
        return list(I2B2_2014_CATEGORIES), SUBTYPE_TO_CATEGORY
    if name == "opf8":
        return list(OPF_LABELS), I2B2_TO_OPF
    if name == "binary":
        return ["PHI"], dict.fromkeys(I2B2_2014_SUBTYPES, "PHI")
    raise ValueError(f"unknown label_space {name!r}")


def tag_list(labels: Sequence[str], scheme: str) -> list[str]:
    if scheme == "bioes":
        return bioes_tags(tuple(labels))
    if scheme == "bio":
        return ["O"] + [f"{p}-{lab}" for lab in labels for p in ("B", "I")]
    raise ValueError(f"unknown scheme {scheme!r}")


def map_gold(doc: Doc, mapping: Mapping[str, str | None] | None) -> list[Span]:
    if mapping is None:
        return list(doc.spans)
    out = []
    for s in doc.spans:
        lab = mapping.get(s.label, s.label)
        if lab is not None:
            out.append(Span(s.start, s.end, lab, s.text, s.source, s.score))
    return out


def special_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """(prefix ids, suffix ids) the tokenizer adds around a sequence, found by tokenizing one word with and without
    special tokens (works across transformers versions; e.g. BERT -> ([CLS], [SEP]), GPT-style -> ([], [EOS]))."""
    bare = tokenizer("x", add_special_tokens=False)["input_ids"]
    full = tokenizer("x", add_special_tokens=True)["input_ids"]
    if not bare:
        raise ValueError("tokenizer produced no tokens for probe text")
    for i in range(len(full) - len(bare) + 1):
        if full[i : i + len(bare)] == bare:
            return list(full[:i]), list(full[i + len(bare) :])
    raise ValueError("could not locate the probe tokens inside the special-token template")


def build_examples(
    docs: Iterable[Doc],
    tokenizer: Any,
    label2id: Mapping[str, int],
    mapping: Mapping[str, str | None] | None,
    max_length: int,
    stride: int,
    scheme: str,
) -> list[dict[str, list[int]]]:
    """Tokenize documents into overlapping windows with aligned tag ids (special tokens -> IGNORE_INDEX)."""
    examples: list[dict[str, list[int]]] = []
    prefix_ids, suffix_ids = special_token_ids(tokenizer)
    body = max_length - len(prefix_ids) - len(suffix_ids)
    for doc in docs:
        enc = tokenizer(doc.text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        ids = enc["input_ids"]
        tags = spans_to_token_labels(map_gold(doc, mapping), offsets, scheme=scheme)
        for ws, we in chunk_windows(len(ids), body, stride):
            w_ids = prefix_ids + ids[ws:we] + suffix_ids
            # labels aligned to the window, IGNORE_INDEX on the added special tokens
            core = tags_to_ids(tags[ws:we], offsets[ws:we], dict(label2id))
            labels = [IGNORE_INDEX] * len(prefix_ids) + core + [IGNORE_INDEX] * len(suffix_ids)
            assert len(labels) == len(w_ids), (len(labels), len(w_ids))
            examples.append({"input_ids": w_ids, "attention_mask": [1] * len(w_ids), "labels": labels})
    return examples


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def train(config: dict[str, Any], seed: int | None = None, max_steps: int | None = None) -> Path:
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    labels, mapping = label_space(config.get("label_space", "i2b2_28"))
    scheme = config.get("scheme", "bioes")
    tags = tag_list(labels, scheme)
    label2id = {t: i for i, t in enumerate(tags)}
    id2label = dict(enumerate(tags))
    seed = seed if seed is not None else int((config.get("seeds") or [42])[0])
    out_dir = Path(config["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer") or config["base_model"], use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        config["base_model"], num_labels=len(tags), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )
    max_length = int(config.get("max_length") or min(getattr(tokenizer, "model_max_length", 512), 512))
    stride = int(config.get("stride", 64))
    train_ex = build_examples(
        read_jsonl(config["train_jsonl"]), tokenizer, label2id, mapping, max_length, stride, scheme
    )
    dev_ex = (
        build_examples(read_jsonl(config["dev_jsonl"]), tokenizer, label2id, mapping, max_length, stride, scheme)
        if config.get("dev_jsonl")
        else None
    )

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict[str, list[int]]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> dict[str, list[int]]:
            return self.rows[i]

    precision = config.get("precision", "fp32")
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        num_train_epochs=float(config.get("epochs", 3)),
        per_device_train_batch_size=int(config.get("per_device_batch_size", 8)),
        per_device_eval_batch_size=int(config.get("per_device_batch_size", 8)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation", 1)),
        bf16=(precision == "bf16" and use_cuda),
        fp16=(precision == "fp16" and use_cuda),
        eval_strategy="epoch" if dev_ex else "no",
        save_strategy="no",
        logging_steps=int(config.get("logging_steps", 50)),
        report_to=[],
        seed=seed,
        max_steps=max_steps if max_steps is not None else -1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ListDataset(train_ex),
        eval_dataset=ListDataset(dev_ex) if dev_ex else None,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_make_compute_metrics(id2label) if dev_ex else None,
    )
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "note_deid_train.json").write_text(
        json.dumps({"config": config, "seed": seed, "labels": labels, "scheme": scheme}, indent=2), encoding="utf-8"
    )
    return out_dir


def _make_compute_metrics(id2label: Mapping[int, str]):
    def compute(eval_pred: Any) -> dict[str, float]:
        import numpy as np

        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true, y_pred = [], []
        for p_row, l_row in zip(preds, label_ids, strict=True):
            t, p = [], []
            for pi, li in zip(p_row, l_row, strict=True):
                if li == IGNORE_INDEX:
                    continue
                t.append(id2label[int(li)])
                p.append(id2label[int(pi)])
            y_true.append(t)
            y_pred.append(p)
        try:
            from seqeval.metrics import f1_score, precision_score, recall_score

            return {
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
            }
        except ImportError:
            n = sum(len(t) for t in y_true)
            correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) for a, b in zip(t, p, strict=True) if a == b)
            return {"token_accuracy": correct / n if n else 0.0}

    return compute


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--output-dir")
    ns = ap.parse_args(argv)
    cfg = load_config(ns.config)
    if ns.output_dir:
        cfg["output_dir"] = ns.output_dir
    out = train(cfg, seed=ns.seed, max_steps=ns.max_steps)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
