# Experiment configs

One YAML per configuration (phase P1+). Each names: benchmark version and split, detectors (LLM alias + prompt/format
ids; TC checkpoint), fusion policy and thresholds, seeds, and the results directory. Runs write
`results/<exp>/<run-id>/{predictions.jsonl, metrics.json, cost.json, config.yaml}` so every number can be recomputed.

Planned: `e1_complementarity.yaml`, `e8_formats.yaml`, `e2_fusion_*.yaml`, `e3_learning_curve.yaml`,
`e9_decision.yaml` (candidate proposers x decision designs D1-D7, needs the `decision` detector type), ...
(see framework-designs.md §4). Planning-stage cost estimates for E9: `python -m note_deid.llm.costmodel`.
