"""Token-classification backend: span/tag encoding with chunking, tag decoding with span scores, a Transformers
detector (``TCDetector``, imported lazily because it needs torch) and a Trainer-based fine-tuning script
(``python -m note_deid.tc.train``).
"""

from note_deid.tc.decode import decode_tags
from note_deid.tc.encode import (
    IGNORE_INDEX,
    chunk_windows,
    merge_window_predictions,
    regex_token_offsets,
    spans_to_token_labels,
    tags_to_ids,
)

__all__ = [
    "IGNORE_INDEX",
    "chunk_windows",
    "decode_tags",
    "merge_window_predictions",
    "regex_token_offsets",
    "spans_to_token_labels",
    "tags_to_ids",
]
