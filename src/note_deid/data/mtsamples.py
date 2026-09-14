"""MTSamples loader (public transcribed sample reports; no real PHI).

Expected input: the CSV distributed as ``mtsamples.csv`` (Kaggle "Medical Transcriptions"), with columns
``description, medical_specialty, sample_name, transcription, keywords``. Check the site's terms before
redistributing derived text; the injection benchmark is distributed as generator + manifest.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

from note_deid.schema import Doc


def load_mtsamples(csv_path: str | Path, min_chars: int = 200) -> Iterator[Doc]:
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            text = (row.get("transcription") or "").strip()
            if len(text) < min_chars:
                continue
            specialty = (row.get("medical_specialty") or "").strip().strip(",").lower().replace(" ", "_")
            doc_id = f"mts-{i:05d}-{hashlib.sha1(text.encode()).hexdigest()[:8]}"
            yield Doc(
                doc_id,
                text,
                [],
                {
                    "source": "mtsamples",
                    "note_type": specialty or "unknown",
                    "sample_name": (row.get("sample_name") or "").strip(),
                },
            )
