#!/bin/bash
set -e

echo "=============================================="
echo "FIM Training Setup (using uv)"
echo "=============================================="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

WORKSPACE="/workspace/fim-training"
mkdir -p $WORKSPACE
cd $WORKSPACE

echo ""
echo "Creating venv with uv..."
uv venv .venv --python 3.11
source .venv/bin/activate

echo ""
echo "Installing packages..."
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
uv pip install tqdm

echo ""
echo "Verifying..."
python -c "
from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
print('All imports OK!')
print(f'Torch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "Run: source .venv/bin/activate && python train_runpod.py"
echo "=============================================="
