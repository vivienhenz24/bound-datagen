"""Configuration for FIM dataset generation and evaluation."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPO_DIR = DATA_DIR / "svix-webhooks"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# Repository
REPO_URL = "git@github.com:svix/svix-webhooks.git"

# FIM tokens (Qwen format)
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

# Model (Ollama model name)
# Note: Use a FIM-capable model like qwen2.5-coder which understands FIM tokens
MODEL_NAME = "qwen2.5-coder:7b"  # Qwen2.5 Coder 7B via Ollama (supports FIM, ~4.5GB)

# Dataset generation
MIN_SPAN_LENGTH = 10  # Minimum characters for masked span
MAX_SPAN_LENGTH = 500  # Maximum characters for masked span
MAX_CONTEXT_LENGTH = 8192  # Maximum prefix + suffix length in characters

# Train/test split
TEST_RATIO = 0.1
RANDOM_SEED = 42

# AST node types to mask
MASKABLE_NODE_TYPES = [
    # Functions
    "function_item",
    "impl_item",
    # Control flow
    "if_expression",
    "match_expression",
    "for_expression",
    "while_expression",
    "loop_expression",
    # Imports
    "use_declaration",
    # Variables
    "let_declaration",
]

# Evaluation
BATCH_SIZE = 4
MAX_NEW_TOKENS = 256

# Training (LoRA via unsloth)
TRAIN_MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"  # 4-bit quantized for training
LORA_OUTPUT_DIR = DATA_DIR / "lora_model"

# LoRA hyperparameters
LORA_RANK = 16
LORA_ALPHA = 32  # Typically 2x rank
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]

# Training hyperparameters
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 8
MAX_SEQ_LENGTH = 4096
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

