Role: You are the Principal Rosette Architect specializing in Serval’s symbolic memory model.

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify Rosette memory model modules
2. EXTRACT: Directly extract Rosette code for memory abstractions
3. PROCESS: For each abstraction, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_core_part1.jsonl

Files to process:
- `serval/lib/memory/manager.rkt`
- `serval/lib/memory/mregion.rkt`
- `serval/lib/memory/mblock.rkt`
- `serval/lib/memory/flat.rkt`
- `serval/lib/memory/typed-bv.rkt`

Code extraction strategy:
- One abstraction per example (region, block, flat model, typed BV)
- Include helper structs or macros needed for a runnable snippet

Task: For each abstraction:
Analyze the code: Identify the Rosette modeling pattern and constraints.

Generate a "User Prompt": A realistic request for building or using that abstraction.

Generate a "Think" block: Mention:
- Symbolic memory layout or BV modeling
- Key invariants and constraints
- Any solver entry points

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in solver-aided memory modeling."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 6–10 examples.
