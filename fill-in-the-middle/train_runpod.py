#!/usr/bin/env python3
"""
LoRA fine-tuning script for FIM task using Unsloth + SFTTrainer.

This script follows the exact training approach from:
https://github.com/prvnsmpth/finetune-code-assistant/blob/master/qwen-finetuning.ipynb

Usage:
    cd /workspace/fim-training
    source .venv/bin/activate
    python train_runpod.py
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

WORKSPACE = Path("/workspace/fim-training")
DATA_DIR = WORKSPACE / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
OUTPUT_DIR = WORKSPACE / "outputs"
LORA_OUTPUT_DIR = DATA_DIR / "lora_model"

# Model config (matching notebook)
max_seq_length = 4096
dtype = None  # Auto detection: Float16 for T4/V100, Bfloat16 for Ampere+
load_in_4bit = True

# Using 7B instead of 14B for 24GB VRAM
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"

# Repo name for FIM format (matching notebook's approach)
REPO_NAME = "svix_webhooks"

# =============================================================================
# Dataset Formatting (matching notebook exactly)
# =============================================================================

def format_train_example(example):
    """
    Format training example using Qwen's FIM tokens.
    
    The notebook format is:
    <|repo_name|>{repo_name}
    <|file_sep|>{filePath}
    <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>
    
    Our data has 'prompt' which already contains <|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>
    and 'completion' which is the middle part.
    """
    # Extract file path from our data
    file_path = example.get('file', 'unknown.rs')
    
    # The prompt already has: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
    # We need to reconstruct to match notebook format
    prompt = example['prompt']
    completion = example['completion']
    
    # Build full text matching notebook format
    text = f"<|repo_name|>{REPO_NAME}\n<|file_sep|>{file_path}\n{prompt}{completion}<|endoftext|>"
    
    return {'text': text}


def format_test_example(example):
    """Format test example (without completion for inference)."""
    file_path = example.get('file', 'unknown.rs')
    prompt = example['prompt']
    
    text = f"<|repo_name|>{REPO_NAME}\n<|file_sep|>{file_path}\n{prompt}"
    
    return {'text': text}


# =============================================================================
# Main Training
# =============================================================================

def main():
    # Check for training data
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found at {TRAIN_FILE}")
        print("Please upload train.jsonl to the data directory.")
        return

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Load Model (matching notebook)
    # ==========================================================================
    print(f"\nLoading model: {MODEL_NAME}")
    print(f"  max_seq_length: {max_seq_length}")
    print(f"  dtype: {dtype} (auto)")
    print(f"  load_in_4bit: {load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # ==========================================================================
    # Configure LoRA (matching notebook)
    # ==========================================================================
    print("\nConfiguring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    # ==========================================================================
    # Load Dataset (matching notebook)
    # ==========================================================================
    print("\nLoading dataset...")
    dataset = load_dataset('json', data_files={'train': str(TRAIN_FILE)})
    train_dataset = dataset['train'].map(format_train_example)
    
    print(f"Num train samples: {train_dataset.num_rows}")
    
    # Show sample
    print("\nSample training text (first 500 chars):")
    print(train_dataset[0]['text'][:500])
    print("...")

    # ==========================================================================
    # Training (matching notebook's SFTTrainer approach)
    # ==========================================================================
    print("\nSetting up SFTTrainer...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            report_to="none",
        ),
    )

    # Print training info
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Training samples: {train_dataset.num_rows}")
    print(f"  Batch size: 2 x 4 = 8 (effective)")
    print(f"  Epochs: 2")
    print(f"  Learning rate: 2e-4")
    print(f"  Max sequence length: {max_seq_length}")
    print("=" * 60 + "\n")

    # Train!
    trainer_stats = trainer.train()

    # ==========================================================================
    # Save Model (matching notebook)
    # ==========================================================================
    print(f"\nSaving LoRA model to {LORA_OUTPUT_DIR}")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Checkpoints: {OUTPUT_DIR}")
    print(f"  LoRA model: {LORA_OUTPUT_DIR}")
    print("=" * 60)

    # Print final stats
    print(f"\nTraining stats:")
    print(f"  Total steps: {trainer_stats.global_step}")
    print(f"  Training loss: {trainer_stats.training_loss:.4f}")


if __name__ == "__main__":
    main()
