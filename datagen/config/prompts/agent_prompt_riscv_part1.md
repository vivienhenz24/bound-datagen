Role: You are the Principal Rosette Architect specializing in RISC-V instruction semantics (base types).

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify RISC-V Rosette interpreter modules for base instruction types
2. EXTRACT: Directly extract Rosette code for instruction types
3. PROCESS: For each type, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_riscv_part1.jsonl

Files to process:
- `serval/riscv/interp/r-type.rkt`
- `serval/riscv/interp/i-type.rkt`
- `serval/riscv/interp/s-type.rkt`
- `serval/riscv/interp/b-type.rkt`
- `serval/riscv/interp/u-type.rkt`
- `serval/riscv/interp/j-type.rkt`
- `serval/riscv/interp/common.rkt`
- `serval/riscv/base.rkt`
- `serval/riscv/decode.rkt`

Code extraction strategy:
- One instruction type per example
- Include any helper definitions needed for runnable context

Task: For each instruction type:
Analyze the code: Identify intent and Rosette constructs used.

Generate a "User Prompt": A realistic Rosette request leading to these semantics.

Generate a "Think" block: Mention:
- Instruction field parsing
- Symbolic register file and memory
- Constraints and solver interaction

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in RISC-V Rosette modeling."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 8–12 examples.
