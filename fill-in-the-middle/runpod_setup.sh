#!/bin/bash
# RunPod setup script for FIM LoRA training
# Usage: bash runpod_setup.sh

set -e

echo "=============================================="
echo "FIM LoRA Training Setup for RunPod"
echo "=============================================="

# Create workspace directory
WORKSPACE="/workspace/fim-training"
mkdir -p $WORKSPACE
cd $WORKSPACE

echo ""
echo "[1/5] Installing system dependencies..."
apt-get update && apt-get install -y git

echo ""
echo "[2/5] Installing Python packages..."
pip install --upgrade pip

# Install PyTorch with CUDA support (should already be installed on RunPod)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install unsloth (optimized for fast training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other dependencies
pip install datasets trl peft accelerate sacrebleu tqdm

echo ""
echo "[3/5] Setting up training files..."

# Check if train.jsonl exists (user should upload this)
if [ ! -f "$WORKSPACE/data/train.jsonl" ]; then
    echo ""
    echo "WARNING: data/train.jsonl not found!"
    echo ""
    echo "Please upload your training data to: $WORKSPACE/data/train.jsonl"
    echo "You can do this via:"
    echo "  1. RunPod file browser"
    echo "  2. scp: scp data/train.jsonl root@<runpod-ip>:$WORKSPACE/data/"
    echo "  3. wget from a URL"
    echo ""
    mkdir -p $WORKSPACE/data
    echo "Created $WORKSPACE/data/ directory. Upload train.jsonl and run this script again."
    exit 1
fi

# Copy training script if not exists
if [ ! -f "$WORKSPACE/train_runpod.py" ]; then
    echo "ERROR: train_runpod.py not found!"
    echo "Please upload train_runpod.py to $WORKSPACE/"
    exit 1
fi

echo ""
echo "[4/5] Checking GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

echo ""
echo "[5/5] Starting training..."
echo "=============================================="

python train_runpod.py

echo ""
echo "=============================================="
echo "Training complete!"
echo "LoRA adapters saved to: $WORKSPACE/data/lora_model/"
echo "=============================================="

