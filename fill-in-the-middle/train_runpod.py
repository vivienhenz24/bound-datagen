#!/usr/bin/env python3
"""
LoRA fine-tuning script optimized for RunPod (RTX 3090/4090 - 24GB VRAM).

Usage:
    1. Upload data/train.jsonl to /workspace/fim-training/data/
    2. Upload this script to /workspace/fim-training/
    3. Run: python train_runpod.py
"""

import json
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

# =============================================================================
# Configuration (optimized for 24GB VRAM)
# =============================================================================

# Paths
WORKSPACE = Path("/workspace/fim-training")
DATA_DIR = WORKSPACE / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
LORA_OUTPUT_DIR = DATA_DIR / "lora_model"

# Model
TRAIN_MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"

# FIM tokens
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

# LoRA hyperparameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]

# Training hyperparameters (optimized for 24GB VRAM)
TRAIN_BATCH_SIZE = 4  # Increased from 1 for faster training
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8
MAX_SEQ_LENGTH = 4096
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

# =============================================================================
# Training Functions
# =============================================================================


def load_training_data(train_file: Path) -> Dataset:
    """Load training data from JSONL file."""
    samples = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} training samples")
    return Dataset.from_list(samples)


def format_fim_sample(sample: dict) -> dict:
    """
    Format a sample for FIM training.
    
    Format: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{completion}<|endoftext|>
    """
    text = sample["prompt"] + sample["completion"] + "<|endoftext|>"
    return {"text": text}


def main() -> None:
    """Main training function."""
    # Check for training data
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found at {TRAIN_FILE}")
        print("Please upload train.jsonl to the data directory.")
        return

    # Create output directory
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model with unsloth
    print(f"\nLoading model: {TRAIN_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAIN_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (will use bfloat16 on Ampere GPUs)
        load_in_4bit=True,
    )

    # Configure LoRA
    print(f"\nConfiguring LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load and format training data
    print("\nLoading training data...")
    dataset = load_training_data(TRAIN_FILE)
    dataset = dataset.map(format_fim_sample, remove_columns=dataset.column_names)
    print(f"Formatted {len(dataset)} samples for training")

    # Sample preview
    print("\nSample training text (first 500 chars):")
    print(dataset[0]["text"][:500] + "...")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,  # Use bfloat16 on Ampere GPUs (3090/4090)
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        dataloader_num_workers=4,  # Faster data loading
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    # Print training info
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Model: {TRAIN_MODEL_NAME}")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  Batch size: {TRAIN_BATCH_SIZE} (effective: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Training samples: {len(dataset)}")
    print("=" * 60 + "\n")

    # Train
    trainer.train()

    # Save LoRA adapters
    print(f"\nSaving LoRA adapters to {LORA_OUTPUT_DIR}")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    # Also save as merged model for easy inference (optional)
    print("\nSaving merged model for inference...")
    model.save_pretrained_merged(
        LORA_OUTPUT_DIR / "merged",
        tokenizer,
        save_method="merged_16bit",
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  LoRA adapters: {LORA_OUTPUT_DIR}")
    print(f"  Merged model: {LORA_OUTPUT_DIR / 'merged'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

