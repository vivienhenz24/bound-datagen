#!/usr/bin/env python3
"""
Simple LoRA fine-tuning script for RunPod - no unsloth, just standard HF stack.
Works reliably with transformers + peft + trl.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# =============================================================================
# Configuration
# =============================================================================

WORKSPACE = Path("/workspace/fim-training")
DATA_DIR = WORKSPACE / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
LORA_OUTPUT_DIR = DATA_DIR / "lora_model"

# Model - use standard HF model (not unsloth version)
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B"

# LoRA config
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training config
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8
MAX_SEQ_LENGTH = 4096
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4


def load_training_data(train_file: Path) -> Dataset:
    samples = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} training samples")
    return Dataset.from_list(samples)


def format_fim_sample(sample: dict) -> dict:
    text = sample["prompt"] + sample["completion"] + "<|endoftext|>"
    return {"text": text}


def main():
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found at {TRAIN_FILE}")
        return

    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("\nLoading training data...")
    dataset = load_training_data(TRAIN_FILE)
    dataset = dataset.map(format_fim_sample, remove_columns=dataset.column_names)
    print(f"Formatted {len(dataset)} samples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  Batch size: {TRAIN_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Samples: {len(dataset)}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save
    print(f"\nSaving LoRA adapters to {LORA_OUTPUT_DIR}")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
