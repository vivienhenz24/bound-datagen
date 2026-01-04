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
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-1.5B with Unsloth.")
    parser.add_argument("--data", default="finetune-data.jsonl", help="Path to JSONL dataset.")
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-1.5B",
        help="Hugging Face model id for Qwen3 1.5B.",
    )
    parser.add_argument("--output-dir", default="output/qwen3-1.5b-unsloth", help="Output directory.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

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

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        warmup_steps=20,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        report_to=[],
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
