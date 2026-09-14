#!/usr/bin/env bash
# Development environment on Apple Silicon (PyTorch MPS). Run from the repository root.
set -euo pipefail

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[llm,finetune,data,dev]"

# OpenAI Privacy Filter CLI (`opf`): editable install from the upstream repo (Apache-2.0).
mkdir -p third_party
[ -d third_party/privacy-filter ] || git clone https://github.com/openai/privacy-filter.git third_party/privacy-filter
pip install -e third_party/privacy-filter

python - <<'PY'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
PY
echo "OPF smoke test:  opf --help   (use --device cpu if MPS is unsupported for the MoE kernels)"
pytest -q
