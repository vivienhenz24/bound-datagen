# Fill-in-the-Middle Dataset Generator

Generate FIM training data from the svix-webhooks Rust codebase for fine-tuning Qwen3 1.7B Coder.

## Overview

This pipeline:
1. Clones the svix-webhooks repository
2. Parses Rust source files using tree-sitter
3. Extracts maskable AST nodes (functions, control flow, imports, variables)
4. Generates FIM samples in Qwen format
5. Splits data into train/test sets
6. Evaluates base model accuracy before fine-tuning

## Installation

```bash
pip install -r requirements.txt
```

**Note:** You also need [Ollama](https://ollama.ai) installed and running for model inference.

## Usage

### 1. Clone the repository

```bash
python clone_repo.py
```

### 2. Generate the dataset

```bash
python dataset.py
```

This will create `data/train.jsonl` and `data/test.jsonl`.

### 3. Evaluate base model

```bash
python evaluate.py
```

This computes exact match accuracy and BLEU score on the test set.

### 4. Train with LoRA (optional)

```bash
pip install -r requirements.txt  # Includes unsloth dependencies
python train.py
```

This fine-tunes Qwen2.5-Coder-7B with LoRA on the FIM dataset. Training parameters:
- LoRA rank: 16
- Batch size: 1 (effective: 8 with gradient accumulation)
- Epochs: 2
- Learning rate: 2e-4

LoRA adapters will be saved to `data/lora_model/`.

## FIM Sample Format

Each sample uses Qwen's FIM tokens:

```json
{
  "prompt": "<|fim_prefix|>fn main() {\n    let x = 5;\n<|fim_suffix|>\n    println!(\"{}\", y);\n}<|fim_middle|>",
  "completion": "    let y = x * 2;",
  "file": "src/main.rs",
  "node_type": "let_declaration",
  "line_range": [3, 3]
}
```

## Masked Node Types

- **Functions**: `function_item`, `impl_item`
- **Control flow**: `if_expression`, `match_expression`, `for_expression`, `while_expression`, `loop_expression`
- **Imports**: `use_declaration`
- **Variables**: `let_declaration`

## Configuration

See `config.py` for all configurable parameters including:
- Min/max span lengths
- Context window size
- Train/test split ratio
- Model name

