Role: You are the Principal Rosette Architect specializing in x86 instruction semantics (shift/rotate/move/branch).

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify x86 Rosette interpreter modules for shifts, rotates, control flow, and data movement
2. EXTRACT: Directly extract Rosette code for specific x86 ops
3. PROCESS: For each op, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_x86_part2.jsonl

Files to process:
- `serval/x86/interp/shift.rkt`
- `serval/x86/interp/rotate.rkt`
- `serval/x86/interp/shld.rkt`
- `serval/x86/interp/shrd.rkt`
- `serval/x86/interp/mov.rkt`
- `serval/x86/interp/movzx.rkt`
- `serval/x86/interp/xchg.rkt`
- `serval/x86/interp/bswap.rkt`
- `serval/x86/interp/jcc.rkt`
- `serval/x86/interp/jmp.rkt`
- `serval/x86/interp/call.rkt`
- `serval/x86/interp/ret.rkt`
- `serval/x86/interp/push.rkt`
- `serval/x86/interp/pop.rkt`
- `serval/x86/base.rkt`
- `serval/x86/register.rkt`

Code extraction strategy:
- One opcode family per example
- Include symbolic state and flags where used

Task: For each opcode family:
Analyze the code: Identify opcode intent and Rosette constructs used.

Generate a "User Prompt": A realistic Rosette request for that semantic.

Generate a "Think" block: Mention:
- Flag semantics and bitvector widths
- Symbolic registers/memory
- Constraint or solver usage

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in x86 Rosette modeling."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 10–15 examples.
