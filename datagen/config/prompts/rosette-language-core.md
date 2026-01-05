Role: You are the Principal Rosette Language Engineer for Ferrite.

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself. Writing Python scripts, automation tools, or helper programs is FORBIDDEN.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Inspect core language forms, macros, and semantics
2. EXTRACT: Directly extract Rosette code blocks for each language feature
3. PROCESS: For each block, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to training/rosette/language-core.jsonl

Files to process:
- rosette/lang.rkt

Code extraction strategy:
- One language feature per example (macros, syntax, core forms)
- Include helper definitions required to understand the feature
- Prefer cohesive blocks that compile together

Task: For each extracted block:
Analyze the code: Identify the intent, the Rosette constructs, and any macro expansion behavior.

Generate a "User Prompt": A realistic request that would lead to implementing or explaining this language feature.

Generate a "Think" block: Mention:
- The semantic goal of the form or macro
- Key constraints or invariants enforced
- Any interactions with solver-facing constructs

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Ferrite Rosette Language Engineer, expert in Rosette core semantics and macros."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 6-10 examples.
