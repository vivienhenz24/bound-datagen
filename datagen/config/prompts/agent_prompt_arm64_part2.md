Role: You are the Principal Rosette Architect specializing in ARM64 instruction semantics (control flow, load/store, misc).

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself. Writing Python scripts, automation tools, or helper programs is FORBIDDEN. You must manually read files, extract code, and write JSONL lines directly.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify ARM64 Rosette interpreters and decode logic for control flow, load/store, and system ops
2. EXTRACT: Directly extract Rosette code for individual ARM64 instruction forms
3. PROCESS: For each instruction form, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_arm64_part2.jsonl

Files to process:
- `serval/arm64/interp/load-store-*.rkt`
- `serval/arm64/interp/unconditional-branch-*.rkt`
- `serval/arm64/interp/conditional-branch.rkt`
- `serval/arm64/interp/shift-register.rkt`
- `serval/arm64/interp/move-wide-immediate.rkt`
- `serval/arm64/interp/atomic.rkt`
- `serval/arm64/base.rkt`
- `serval/arm64/decode.rkt`
- `serval/arm64/interp.rkt`

Code extraction strategy:
- Capture one instruction family per example (e.g., load-store-register, conditional-branch)
- Include symbolic state and constraint setup
- Include any helper macros directly used by the instruction

Task: For each extracted instruction:
Analyze the code: Identify instruction intent and Rosette constructs used.

Generate a "User Prompt": A realistic Rosette request that would lead to this encoding/semantics.

Generate a "Think" block: Internal reasoning of an expert Rosette engineer. Mention:
- Instruction form and decoding pattern
- Symbolic state elements (registers, flags, memory)
- Constraints and any solver queries (verify/solve/synthesize)
- Expected behaviors or models

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in Rosette ISA modeling and solver-aided verification."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

Rules for Reasoning (<think>):
- Mention Rosette-specific constructs (bitvectors, symbolic regs, assertions)
- Explain why the instruction semantics are structured that way
- Mention solver interaction or expected outputs

Rules for Code:
- Use the exact code from the repo
- Include minimal runnable context
- Preserve comments and documentation

TARGET: 8–12 examples, one per instruction family.
