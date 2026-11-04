# Experiment configs

One YAML per configuration (phase P1+). Each names: benchmark version and split, detectors (LLM alias + prompt/format
ids; TC checkpoint), fusion policy and thresholds, seeds, and the results directory. `python -m note_deid.run
--config <yaml>` writes `results/<exp>/<run-id>/{config.yaml, predictions/<system>.jsonl, metrics.json,
complementarity.json, cost.json, summary.md}` so every number can be recomputed.

Present: `smoke_simulated.yaml` (offline, simulated detectors over the `--demo` benchmark), `e1_complementarity.yaml`,
`e8_formats.yaml`. Planned: `e2_fusion_*.yaml`, `e3_learning_curve.yaml`, ... (see framework-designs.md §4).
