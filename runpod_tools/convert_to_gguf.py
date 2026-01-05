#!/usr/bin/env python3
"""Convert merged HuggingFace model to GGUF format for Ollama import.

This script converts a merged HuggingFace model to GGUF format using llama.cpp,
which is required for architectures not directly supported by Ollama's import
(e.g., Qwen3).

Usage:
    # Install llama.cpp first (see script output for instructions)
    uv run --with transformers python runpod_tools/convert_to_gguf.py --model-dir output_3epochs/qwen3-1.7b-merged --output output_3epochs/qwen3-1.7b.gguf
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger("convert_gguf")


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Logging configured (debug=%s)", debug)


def check_llama_cpp() -> bool:
    """Check if llama.cpp conversion script is available."""
    try:
        # Try to find convert-hf-to-gguf.py
        result = subprocess.run(
            ["which", "convert-hf-to-gguf.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        
        # Try python -m llama_cpp
        result = subprocess.run(
            [sys.executable, "-m", "llama_cpp", "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to GGUF format for Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to merged HuggingFace model directory",
    )
    parser.add_argument(
        "--output",
        help="Output GGUF file path (default: <model-dir>/model.gguf)",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f16", "f32"],
        help="Output type: f16 (default) or f32",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    configure_logging(args.debug)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = model_dir / "model.gguf"

    LOGGER.info("Model directory: %s", model_dir)
    LOGGER.info("Output file: %s", output_file)

    # Check for llama.cpp
    LOGGER.info("Checking for llama.cpp conversion tools...")
    if not check_llama_cpp():
        LOGGER.error("llama.cpp conversion tools not found!")
        LOGGER.error("")
        LOGGER.error("Please install llama.cpp conversion tools:")
        LOGGER.error("")
        LOGGER.error("Option 1: Install llama-cpp-python:")
        LOGGER.error("  uv pip install llama-cpp-python")
        LOGGER.error("")
        LOGGER.error("Option 2: Clone llama.cpp and use its scripts:")
        LOGGER.error("  git clone https://github.com/ggerganov/llama.cpp.git")
        LOGGER.error("  cd llama.cpp")
        LOGGER.error("  python convert-hf-to-gguf.py %s --outfile %s --outtype %s",
                     model_dir, output_file, args.outtype)
        LOGGER.error("")
        LOGGER.error("Or use the llama.cpp Docker image:")
        LOGGER.error("  docker run --rm -v %s:/model ggerganov/llama.cpp:latest python3 convert-hf-to-gguf.py /model --outfile /model/model.gguf --outtype %s",
                     model_dir.absolute(), args.outtype)
        sys.exit(1)

    # Try to use llama.cpp conversion
    LOGGER.info("Converting model to GGUF format...")
    LOGGER.info("This may take several minutes depending on model size...")

    # Try different methods to call the conversion script
    conversion_commands = [
        # Method 1: Direct script
        ["convert-hf-to-gguf.py", str(model_dir), "--outfile", str(output_file), "--outtype", args.outtype],
        # Method 2: Python module
        [sys.executable, "-m", "llama_cpp.convert", str(model_dir), "--outfile", str(output_file), "--outtype", args.outtype],
    ]

    success = False
    for cmd in conversion_commands:
        try:
            LOGGER.info("Trying: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
            )
            success = True
            break
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            LOGGER.error("Conversion failed: %s", e)
            continue

    if not success:
        LOGGER.error("")
        LOGGER.error("Automatic conversion failed. Please run manually:")
        LOGGER.error("")
        LOGGER.error("If you have llama.cpp cloned:")
        LOGGER.error("  python llama.cpp/convert-hf-to-gguf.py %s --outfile %s --outtype %s",
                     model_dir, output_file, args.outtype)
        LOGGER.error("")
        LOGGER.error("Or use Docker:")
        LOGGER.error("  docker run --rm -v %s:/model ggerganov/llama.cpp:latest python3 convert-hf-to-gguf.py /model --outfile /model/model.gguf --outtype %s",
                     model_dir.absolute(), args.outtype)
        sys.exit(1)

    LOGGER.info("Conversion complete! GGUF file saved to: %s", output_file)
    LOGGER.info("")
    LOGGER.info("Now create a Modelfile:")
    LOGGER.info("  echo 'FROM %s' > %s/Modelfile", output_file.absolute(), output_file.parent)
    LOGGER.info("")
    LOGGER.info("Then import to Ollama:")
    LOGGER.info("  cd %s", output_file.parent)
    LOGGER.info("  ollama create qwen3-finetuned -f Modelfile")
    LOGGER.info("")
    LOGGER.info("Run it:")
    LOGGER.info("  ollama run qwen3-finetuned")


if __name__ == "__main__":
    main()


