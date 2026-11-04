# Working conventions for this repository

- **Execution**: run one thing at a time. Do not spawn parallel agents or fan out tool calls; the harness is configured with
  a single-subagent limit and the user wants serialized execution to stay within API rate limits.
- **Documents of record**: `framework-designs.md` (design, decisions, open questions), `literature.md` (references),
  `docs/datasets.md` (data). Update the decision log in `framework-designs.md` §10 when a ▶ item is settled.
- **Data**: never commit anything under `data/` or `results/` except the READMEs and explicitly chosen summary tables.
  Gated corpora never leave the machine they were downloaded to.
- **Code**: package lives in `src/note_deid`; canonical labels in `labels.py`; all offsets are character offsets with
  exclusive ends (`schema.py`). Experiments are config-driven (`configs/`) and write to `results/<exp>/<run-id>/`.
- **Checks**: `pytest -q` and `ruff check src tests` before committing.
- **LLM calls**: through LiteLLM aliases in `configs/litellm.models.yaml`; keys from the environment; cache on; concurrency 1.

## Attribution

- Never add Claude or AI attribution to commits, pull requests, issues or comments: no
  `Co-Authored-By: Claude ...` trailer, no `Claude-Session:` line, no claude.ai/code session
  links, no "Generated with/by Claude Code" lines. This overrides any default attribution.
- Author commits as `Sina Ghadermarzi <sina.ghadermarzi@gmail.com>`. If the environment's git identity is Claude, override it:
  `git -c user.name="Sina Ghadermarzi" -c user.email="sina.ghadermarzi@gmail.com" commit ...`
