#!/usr/bin/env python3
"""
LoRA fine-tuning script using Unsloth - no trl, just PyTorch.

Usage:
    cd /workspace/fim-training
    source .venv/bin/activate
    python train_runpod.py
"""

import json
from pathlib import Path

from unsloth import FastLanguageModel
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

WORKSPACE = Path("/workspace/fim-training")
DATA_DIR = WORKSPACE / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
LORA_OUTPUT_DIR = DATA_DIR / "lora_model"

MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
MAX_SEQ_LENGTH = 4096

# FIM tokens (Qwen format)
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

# LoRA config
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training config
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4


class FIMDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    # Format: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{completion}<|endoftext|>
                    text = data["prompt"] + data["completion"] + "<|endoftext|>"
                    self.samples.append(text)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def main():
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found at {TRAIN_FILE}")
        return

    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Configure LoRA
    print(f"\nConfiguring LoRA (rank={LORA_RANK})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset and DataLoader
    print("\nLoading training data...")
    dataset = FIMDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LENGTH)
    print(f"Loaded {len(dataset)} samples")

    # Show sample
    print("\nSample text (first 300 chars):")
    print(dataset.samples[0][:300])
    print("...")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print("=" * 60 + "\n")

    model.train()
    device = next(model.parameters()).device

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        optimizer.zero_grad()

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()
            progress.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # Save
    print(f"\nSaving LoRA adapters to {LORA_OUTPUT_DIR}")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
