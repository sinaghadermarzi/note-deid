# Experiment configs

One YAML per configuration (phase P1+). Each names: benchmark version and split, detectors (LLM alias + prompt/format
ids; TC checkpoint), fusion policy and thresholds, seeds, and the results directory. Runs write
`results/<exp>/<run-id>/{predictions.jsonl, metrics.json, cost.json, config.yaml}` so every number can be recomputed.

Planned: `e1_complementarity.yaml`, `e8_formats.yaml`, `e2_fusion_*.yaml`, `e3_learning_curve.yaml`, ...
(see framework-designs.md §4).
