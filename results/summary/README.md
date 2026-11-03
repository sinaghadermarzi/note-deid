# results/summary — committed summary tables

Each table is produced by `python -m note_deid.summarize results/<exp>/<run-id> --out results/summary/<exp>` and
carries the `run_id` it came from. Nothing is recomputed: every number is copied from that run's `metrics.json` /
`complementarity.json`.

| Table | Run | What it is | Evidence? |
|---|---|---|---|
| `smoke_simulated.csv`, `smoke_simulated.complementarity.csv` | `smoke_simulated` on the demo `synphi` build (`--demo --repeat 4 --seed 0`: 24 notes, 5 in the test split) | **Pipeline check, not an experimental result.** Both "detectors" are `SimulatedDetector` stand-ins that corrupt the gold spans with configured miss/false-positive/boundary rates (`configs/experiments/smoke_simulated.yaml`); the fusion policies, evaluator, HIPAA view, paired bootstrap and complementarity statistics run for real on that output. | No. It shows that the E1 pipeline runs end to end offline and what its output looks like. |

No table here comes from a real LLM or a trained token classifier yet (see the README status). When E1 runs, its
summary goes in this directory as `e1_complementarity.csv` with the run id, benchmark version and model identifiers.

Columns of `<exp>.csv`: one row per system x matching mode x label level x view (`full` = all 28 i2b2 subtypes;
`hipaa` = the HIPAA Safe-Harbor subset with the age >= 90 and no-bare-year rules); `delta_recall*` is the paired
bootstrap of recall against the run's reference system (`eval.reference`, here `tc`). Columns of
`<exp>.complementarity.csv`: one row per detector pair and mode, with oracle-union / intersection recall, the
false-negative overlap counts and Jaccard, conditional miss probabilities, McNemar p and kappa.
