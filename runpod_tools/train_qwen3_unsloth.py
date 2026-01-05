#!/usr/bin/env python3
"""Fine-tune Qwen3-1.5B with Unsloth on a JSONL chat dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from unsloth import FastModel

from datasets import load_dataset
from huggingface_hub import login
from transformers import TrainingArguments
from trl import SFTTrainer


LOGGER = logging.getLogger("runpod_finetune")


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Logging configured (debug=%s)", debug)


def read_jsonl_sample(path: Path, num_lines: int = 1) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(num_lines):
            line = handle.readline()
            if not line:
                break
            samples.append(json.loads(line))
    return samples


def format_chat_dataset(dataset, tokenizer, max_seq_length: int):
    def format_example(example: Dict) -> Dict[str, str]:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting chat messages with tokenizer chat template",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 with Unsloth.")
    parser.add_argument("--data", default=None, help="Path to JSONL dataset.")
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model id for Qwen3.",
    )
    parser.add_argument("--output-dir", default="output/qwen3-1.7b-unsloth", help="Output directory.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (Unsloth default: 4).")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Override total training steps (takes precedence over epochs if > 0).",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--logging-steps", type=int, default=1, help="Logging interval in steps (Unsloth default: 1).")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint save interval in steps.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps (Unsloth default: 10).")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Warmup ratio if steps not set.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.data is None:
        print("\nChoose dataset for training:")
        print("1) finetune-data.jsonl (with reasoning)")
        print("2) finetune-data_no_reasoning.jsonl (removed reasoning)")
        print("3) datagen/output/merged_rosette_final.jsonl (Rosette/Serval/Ocelot/Cosette/Quivela/Ferrite)")
        choice = input("\nEnter choice [1/2/3, default 1]: ").strip()

        if choice == "2":
            args.data = "finetune-data_no_reasoning.jsonl"
        elif choice == "3":
            args.data = "datagen/output/merged_rosette_final.jsonl"
        else:
            args.data = "finetune-data.jsonl"

        print(f"Selected dataset: {args.data}")

    if args.model is None:
        print("\nChoose model for training:")
        print("1) Qwen/Qwen3-1.7B (default)")
        print("2) Qwen/Qwen3-8B")
        print("3) Qwen/Qwen3-0.6B")
        choice = input("\nEnter choice [1/2/3, default 1]: ").strip()

        if choice == "2":
            args.model = "Qwen/Qwen3-8B"
        elif choice == "3":
            args.model = "Qwen/Qwen3-0.6B"
        else:
            args.model = "Qwen/Qwen3-1.7B"

        print(f"Selected model: {args.model}\n")

    # Dynamic output directory and hyperparameter adjustment
    # Use model and dataset characteristics to set a better default output path if not explicitly provided
    if args.output_dir == "output/qwen3-1.7b-unsloth":
        if "8B" in str(args.model):
            model_slug = "qwen3-8b"
        elif "0.6B" in str(args.model):
            model_slug = "qwen3-0.6b"
        else:
            model_slug = "qwen3-1.7b"

        # Determine data slug based on dataset
        if "merged_rosette_final" in str(args.data):
            data_slug = "rosette"
        elif "no_reasoning" in str(args.data):
            data_slug = "no-think"
        else:
            data_slug = "think"

        args.output_dir = f"output/{model_slug}-{data_slug}"
        print(f"Using automatic output directory: {args.output_dir}")

    # For 8B model, optimize hyperparameters for stability and knowledge preservation
    if "8B" in str(args.model):
        if args.grad_accum == 4:
            args.grad_accum = 8
            print("Increasing gradient accumulation to 8 for 8B model stability.")
        
        if args.learning_rate == 2e-4:
            args.learning_rate = 5e-5
            print(f"Adjusted learning rate to {args.learning_rate} for 8B model.")
            
        if args.epochs == 1 or args.epochs == 5: # If using default or the common setup default
            args.epochs = 3
            print(f"Adjusted epochs to {args.epochs} for 8B model to prevent overfitting.")

    # For 1.7B model, optimize to prevent "catastrophic forgetting" and looping
    if "1.7B" in str(args.model):
        if args.batch_size == 2:
            args.batch_size = 8
            print("Increasing batch size to 8 for the RTX 5090.")

        if args.grad_accum == 4:
            args.grad_accum = 1
            print("Reducing gradient accumulation to 1 for 1.7B model stability.")

        if args.learning_rate == 2e-4:
            args.learning_rate = 5e-5
            print(f"Adjusted learning rate to {args.learning_rate} for 1.7B model.")

        if args.epochs == 1 or args.epochs == 5: # If using default or the common setup default
            args.epochs = 2
            print(f"Adjusted epochs to {args.epochs} for 1.7B model stability.")

    # For 0.6B model, optimize for fast training with good convergence
    if "0.6B" in str(args.model):
        if args.batch_size == 2:
            args.batch_size = 16
            print("Increasing batch size to 16 for 0.6B model (fits easily on RTX 5090).")

        if args.grad_accum == 4:
            args.grad_accum = 1
            print("Reducing gradient accumulation to 1 for 0.6B model.")

        if args.learning_rate == 2e-4:
            args.learning_rate = 1e-4
            print(f"Adjusted learning rate to {args.learning_rate} for 0.6B model.")

        if args.epochs == 1 or args.epochs == 5: # If using default or the common setup default
            args.epochs = 3
            print(f"Adjusted epochs to {args.epochs} for 0.6B model (can handle more epochs).")

    configure_logging(args.debug)

    LOGGER.debug("Args: %s", args)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        LOGGER.info("Logging into Hugging Face using HF_TOKEN.")
        login(token=hf_token)
    else:
        LOGGER.warning("HF_TOKEN not set; assuming model is publicly accessible.")

    LOGGER.info("Loading model and tokenizer: %s", args.model)
    LOGGER.debug("Max seq length: %s", args.max_seq_length)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=hf_token,
    )

    LOGGER.info("Configuring LoRA adapters.")
    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    LOGGER.info("Loading dataset from %s", data_path)
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    LOGGER.info("Dataset size: %d", len(dataset))
    sample = read_jsonl_sample(data_path)
    LOGGER.debug("Dataset sample: %s", sample)

    if "messages" not in dataset.column_names:
        raise ValueError("Expected 'messages' field in dataset JSONL.")

    formatted = format_chat_dataset(dataset, tokenizer, args.max_seq_length)
    LOGGER.info("Formatted dataset size: %d", len(formatted))
    try:
        LOGGER.debug("Formatted sample: %s", formatted[0]["text"][:500])
    except Exception:
        LOGGER.debug("Unable to render formatted sample.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use warmup_steps if provided, otherwise use warmup_ratio
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else 0
    warmup_ratio = args.warmup_ratio if warmup_steps == 0 else 0.0

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        report_to=[],
        seed=3407,  # Unsloth default seed for reproducibility
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        packing=False,
    )

    LOGGER.info("Starting training...")
    LOGGER.debug("Training args: %s", training_args)
    trainer.train()
    LOGGER.info("Training complete, saving model to %s", output_dir)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
