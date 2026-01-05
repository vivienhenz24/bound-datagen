Role: You are the Principal Rosette Architect specializing in LLVM Rosette integration (C API + CLI).

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify Rosette LLVM integration points in C API and CLI
2. EXTRACT: Directly extract code handling C API bindings and CLI behavior
3. PROCESS: For each feature, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_llvm_part2.jsonl

Files to process:
- `serval/llvm/capi/core.rkt`
- `serval/llvm/capi/irreader.rkt`
- `serval/llvm/capi/target.rkt`
- `serval/bin/serval-llvm.rkt`

Code extraction strategy:
- One LLVM-related feature per example (C API binding or CLI flow)
- Include any Rosette symbolic elements used for LLVM modeling

Task: For each feature:
Analyze the code: Identify LLVM integration and Rosette constructs used.

Generate a "User Prompt": A realistic request for Rosette+LLVM behavior.

Generate a "Think" block: Mention:
- LLVM constructs being handled
- Rosette data model for IR
- Constraints/solver interaction if present

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in LLVM modeling with Rosette."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 4–6 examples.
