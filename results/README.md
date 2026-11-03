# results/ (gitignored except `summary/`)

`python -m note_deid.run` writes `results/<exp>/<run-id>/{config.yaml, predictions/*.jsonl, metrics.json,
complementarity.json, cost.json, summary.md}` so every number can be recomputed from the run directory. Run
directories are never committed.

Summary tables for the paper are flattened with `python -m note_deid.summarize results/<exp>/<run-id> --out
results/summary/<exp>` and committed explicitly under `results/summary/` (see its README for what each table is and
whether it is evidence).
