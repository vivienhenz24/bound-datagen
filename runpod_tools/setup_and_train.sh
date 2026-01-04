#!/usr/bin/env bash
set -euo pipefail
set -x

echo "[runpod] Starting setup..."
date
pwd

if ! command -v uv >/dev/null 2>&1; then
  echo "[runpod] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
  uv --version || true
fi

if [ ! -d ".venv" ]; then
  echo "[runpod] Initializing uv venv..."
  uv venv
fi

PYTHON_BIN=".venv/bin/python"

echo "[runpod] Installing Python dependencies..."
uv pip install --python "$PYTHON_BIN" -r runpod_tools/requirements.txt
uv pip list --python "$PYTHON_BIN" | head -n 50 || true

echo "[runpod] Verifying GPU availability..."
nvidia-smi || true
nvidia-smi -L || true

echo "[runpod] Launching training..."
"$PYTHON_BIN" runpod_tools/train_qwen3_unsloth.py \
  --data finetune-data.jsonl \
  --output-dir output/qwen3-1.7b-unsloth \
  --debug
