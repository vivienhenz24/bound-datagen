#!/usr/bin/env python3
"""Merge LoRA adapter with base model and prepare for Ollama import.

This script merges a fine-tuned LoRA adapter with the base Qwen3 model,
creating a full merged model that can be imported into Ollama.

Usage:
    # Using uv (recommended):
    uv run --with transformers --with peft --with huggingface-hub python merge_lora_for_ollama.py --adapter-dir output/qwen3-1.7b-unsloth --output-dir output/qwen3-1.7b-merged
    
    # Or install dependencies first:
    uv pip install transformers peft huggingface-hub
    python merge_lora_for_ollama.py --adapter-dir output/qwen3-1.7b-unsloth --output-dir output/qwen3-1.7b-merged

After running this script:
    Qwen3 is not in Ollama's supported architectures for direct import.
    Convert to GGUF format first:
       python runpod_tools/convert_to_gguf.py --model-dir output/qwen3-1.7b-merged
    
    Then create a Modelfile pointing to the .gguf file and import to Ollama.
    
    Alternatively, test the merged model with transformers:
       uv run --with transformers --with torch python runpod_tools/test_merged_model.py --model-dir output/qwen3-1.7b-merged
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login


LOGGER = logging.getLogger("merge_lora")


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Logging configured (debug=%s)", debug)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model for Ollama import.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--adapter-dir",
        default="output/qwen3-1.7b-unsloth",
        help="Path to directory containing LoRA adapter (default: output/qwen3-1.7b-unsloth)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-1.7B",
        help="Base model name or path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/qwen3-1.7b-merged",
        help="Output directory for merged model (default: output/qwen3-1.7b-merged)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    configure_logging(args.debug)

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Adapter config not found: {adapter_config}. Is this a valid LoRA adapter directory?"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        LOGGER.info("Logging into Hugging Face using HF_TOKEN.")
        login(token=hf_token)
    else:
        LOGGER.warning("HF_TOKEN not set; assuming model is publicly accessible.")

    LOGGER.info("Loading base model: %s", args.base_model)
    LOGGER.info("Loading adapter from: %s", adapter_dir)
    LOGGER.info("Output directory: %s", output_dir)

    # Load base model in full precision (not quantized) for proper merging
    LOGGER.info("Loading base model in full precision...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        token=hf_token,
        trust_remote_code=True,
    )

    # Load the LoRA adapter
    LOGGER.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    # Merge and unload the adapter into the base model
    LOGGER.info("Merging adapter with base model...")
    model = model.merge_and_unload()

    # Save the merged model
    LOGGER.info("Saving merged model to %s", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Create Modelfile for Ollama
    modelfile_path = output_dir / "Modelfile"
    LOGGER.info("Creating Modelfile at %s", modelfile_path)
    
    modelfile_content = f"""FROM {output_dir.absolute()}

# Fine-tuned Qwen3-1.7B model merged from LoRA adapter
# Model directory: {output_dir.absolute()}
"""
    
    with modelfile_path.open("w") as f:
        f.write(modelfile_content)

    LOGGER.info("Done! Merged model saved to %s", output_dir)
    LOGGER.info("")
    LOGGER.info("NOTE: Qwen3 is not in Ollama's supported architectures for direct import.")
    LOGGER.info("You need to convert to GGUF format first:")
    LOGGER.info("  python runpod_tools/convert_to_gguf.py --model-dir %s", output_dir)
    LOGGER.info("")
    LOGGER.info("Alternatively, test your model directly with transformers:")
    LOGGER.info("  uv run --with transformers --with torch python runpod_tools/test_merged_model.py --model-dir %s", output_dir)


if __name__ == "__main__":
    main()

