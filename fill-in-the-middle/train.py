#!/usr/bin/env python3
"""LoRA fine-tuning script using unsloth for FIM task."""

import json
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

from config import (
    TRAIN_FILE,
    LORA_OUTPUT_DIR,
    TRAIN_MODEL_NAME,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    LEARNING_RATE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIM_MIDDLE,
)


def load_training_data(train_file: Path) -> Dataset:
    """
    Load training data from JSONL file.

    Args:
        train_file: Path to the training JSONL file

    Returns:
        HuggingFace Dataset
    """
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

    The format follows Qwen's FIM convention:
    <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{completion}<|endoftext|>

    Args:
        sample: Dictionary with 'prompt' and 'completion' keys

    Returns:
        Dictionary with 'text' key containing the formatted sample
    """
    # The prompt already contains: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
    # We just need to append the completion and end token
    text = sample["prompt"] + sample["completion"] + "<|endoftext|>"
    return {"text": text}


def main() -> None:
    """Main training function."""
    # Check for training data
    if not TRAIN_FILE.exists():
        print(f"Training file not found at {TRAIN_FILE}")
        print("Run dataset.py first to generate the dataset.")
        return

    # Load model and tokenizer with unsloth
    print(f"Loading model: {TRAIN_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAIN_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Configure LoRA
    print(f"Configuring LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
    )

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load and format training data
    print("Loading training data...")
    dataset = load_training_data(TRAIN_FILE)

    # Format samples for FIM
    dataset = dataset.map(format_fim_sample, remove_columns=dataset.column_names)
    print(f"Formatted {len(dataset)} samples for training")

    # Sample preview
    print("\nSample training text (first 500 chars):")
    print(dataset[0]["text"][:500] + "...")

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=not model.is_bfloat16_enabled,
        bf16=model.is_bfloat16_enabled,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_8bit",  # Memory-efficient optimizer
        seed=42,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,  # Don't pack multiple samples into one sequence
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Model: {TRAIN_MODEL_NAME}")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  Batch size: {TRAIN_BATCH_SIZE} (effective: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save the LoRA model
    print(f"\nSaving LoRA adapters to {LORA_OUTPUT_DIR}")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapters saved to: {LORA_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

