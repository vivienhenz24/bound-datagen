#!/bin/bash
# RunPod setup script for Unsloth LoRA training
set -e

echo "=============================================="
echo "FIM Training Setup"
echo "=============================================="

WORKSPACE="/workspace/fim-training"
mkdir -p $WORKSPACE
cd $WORKSPACE

# Remove old venv for clean install
rm -rf .venv

echo ""
echo "Step 1: Setting up uv..."
pip install --upgrade pip
pip install uv

echo ""
echo "Step 2: Creating fresh venv with Python 3.11..."
uv venv .venv --python 3.11
source .venv/bin/activate

echo ""
echo "Step 3: Installing Unsloth (auto-installs correct torch + CUDA)..."
uv pip install unsloth

echo ""
echo "Step 4: Installing other required packages..."
uv pip install datasets tqdm

echo ""
echo "Verifying installation..."
python -c "
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
import torch
print('All imports OK!')
print(f'Torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To train:"
echo "  cd /workspace/fim-training"
echo "  source .venv/bin/activate"
echo "  python train_runpod.py"
echo "=============================================="
