#!/usr/bin/env bash
# Environment on a 4x A100 80GB node. Run from the repository root. Adjust module/conda lines to the cluster.
set -euo pipefail

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
# CUDA build of torch: pick the wheel index matching the node's driver, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[llm,finetune,data,dev]"

mkdir -p third_party
[ -d third_party/privacy-filter ] || git clone https://github.com/openai/privacy-filter.git third_party/privacy-filter
pip install -e third_party/privacy-filter

nvidia-smi --query-gpu=name,memory.total --format=csv
pytest -q

cat <<'TXT'
Launch templates (phase P1+):
  # token-classifier fine-tuning, 4 GPUs (DDP)
  torchrun --nproc_per_node=4 -m note_deid.tc.train --config configs/train/opf_finetune.yaml
  # optional local LLM behind the `llm-local` alias
  vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000
TXT
