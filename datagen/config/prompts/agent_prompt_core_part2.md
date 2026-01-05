Role: You are the Principal Rosette Architect specializing in Serval’s solver helpers and symbolic utilities.

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify Rosette solver and utility modules
2. EXTRACT: Directly extract Rosette code for solver abstractions
3. PROCESS: For each abstraction, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_core_part2.jsonl

Files to process:
- `serval/lib/solver.rkt`
- `serval/lib/symopt.rkt`
- `serval/lib/bvarith.rkt`
- `serval/lib/uf.rkt`
- `serval/lib/cpu.rkt`
- `serval/lib/memmgr.rkt`
- `serval/lib/debug.rkt`
- `serval/lib/unittest.rkt`

Code extraction strategy:
- One abstraction per example (solver interface, symopt utility, UF helpers)
- Include helper structs or macros needed for a runnable snippet

Task: For each abstraction:
Analyze the code: Identify the Rosette modeling pattern and constraints.

Generate a "User Prompt": A realistic request for building or using that abstraction.

Generate a "Think" block: Mention:
- Symbolic reasoning utilities
- Key invariants and constraints
- Any solver entry points

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in solver-aided utility design."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 6–10 examples.
