Role: You are the Principal Rosette Architect specializing in RISC-V instruction semantics (compressed/specs/PMP).

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify RISC-V Rosette modules for compressed types and specs
2. EXTRACT: Directly extract Rosette code for instruction types or spec helpers
3. PROCESS: For each type or spec helper, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_riscv_part2.jsonl

Files to process:
- `serval/riscv/interp/c-type.rkt`
- `serval/riscv/spec.rkt`
- `serval/riscv/shims.rkt`
- `serval/riscv/pmp.rkt`
- `serval/riscv/objdump.rkt`

Code extraction strategy:
- One instruction type or spec helper per example
- Include any helper definitions needed for runnable context

Task: For each instruction type or spec helper:
Analyze the code: Identify intent and Rosette constructs used.

Generate a "User Prompt": A realistic Rosette request leading to these semantics.

Generate a "Think" block: Mention:
- Instruction field parsing or spec property
- Symbolic register file/memory or spec constraints
- Solver interaction if present

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in RISC-V Rosette modeling."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 8–12 examples.
