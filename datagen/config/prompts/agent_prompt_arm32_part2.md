Role: You are the Principal Rosette Architect specializing in ARM32 instruction semantics (load/store, branch, misc).

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself. Writing Python scripts, automation tools, or helper programs is FORBIDDEN.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify ARM32 Rosette interpreter modules for memory and control flow
2. EXTRACT: Directly extract Rosette code for instruction groups
3. PROCESS: For each group, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_arm32_part2.jsonl

Files to process:
- `serval/arm32/interp/load-store-*.rkt`
- `serval/arm32/interp/branch-immediate.rkt`
- `serval/arm32/interp/extend-and-add.rkt`
- `serval/arm32/interp/reverse.rkt`
- `serval/arm32/interp/signed-multiply-divide.rkt`
- `serval/arm32/interp/multiply-and-accumulate.rkt`
- `serval/arm32/base.rkt`
- `serval/arm32/interp.rkt`

Code extraction strategy:
- One instruction group per example (e.g., load-store, branch)
- Include symbolic CPU state and helpers needed

Task: For each extracted instruction group:
Analyze the code: Identify intent and Rosette constructs used.

Generate a "User Prompt": A realistic Rosette request leading to this semantics.

Generate a "Think" block: Mention:
- Instruction group and decoding style
- Symbolic regs/flags/memory
- Constraints and solver queries

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in ISA semantics and solver-aided verification."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 8–12 examples.
