#!/usr/bin/env python3
"""Test the merged fine-tuned model directly with transformers.

This script loads and tests your merged model without needing Ollama.

Usage:
    uv run --with transformers --with torch python runpod_tools/test_merged_model.py --model-dir output_3epochs/qwen3-1.7b-merged
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


LOGGER = logging.getLogger("test_model")


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Logging configured (debug=%s)", debug)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test merged fine-tuned model with transformers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="output_3epochs/qwen3-1.7b-merged",
        help="Path to merged model directory (default: output_3epochs/qwen3-1.7b-merged)",
    )
    parser.add_argument(
        "--prompt",
        default="Hello! How can I help you today?",
        help="Prompt to test the model with",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    configure_logging(args.debug)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    LOGGER.info("Loading model from: %s", model_dir)
    
    # Load model and tokenizer
    LOGGER.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    
    LOGGER.info("Loading model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    
    LOGGER.info("Model loaded successfully!")
    LOGGER.info("Testing with prompt: %s", args.prompt)
    
    # Tokenize input
    messages = [
        {"role": "user", "content": args.prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate response
    LOGGER.info("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print("PROMPT:")
    print(args.prompt)
    print("\n" + "="*80)
    print("RESPONSE:")
    print(response)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


