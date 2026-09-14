"""Asclepius synthetic clinical notes loader (HF ``starmpcc/Asclepius-Synthetic-Clinical-Notes``, CC-BY-NC-SA 4.0).

The dataset repeats each note across several instruction tasks; notes are de-duplicated by content.
Requires the ``datasets`` package (``pip install -e ".[data]"``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from note_deid.schema import Doc

DATASET = "starmpcc/Asclepius-Synthetic-Clinical-Notes"


def load_asclepius(limit: int | None = None, split: str = "train", min_chars: int = 200) -> Iterator[Doc]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split, streaming=True)
    seen: set[str] = set()
    n = 0
    for row in ds:
        text = (row.get("note") or "").strip()
        if len(text) < min_chars:
            continue
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        if h in seen:
            continue
        seen.add(h)
        yield Doc(f"asc-{h}", text, [], {"source": "asclepius", "note_type": "synthetic_discharge_summary"})
        n += 1
        if limit is not None and n >= limit:
            return
