Role: You are the Principal Rosette Architect specializing in program verification specs (ubsan/sha256).

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Identify Rosette specs and verification code in ubsan/sha256
2. EXTRACT: Directly extract spec code and solver queries
3. PROCESS: For each spec pattern, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to rosette_specs_part2.jsonl

Files to process:
- `serval/ubsan.rkt`
- `serval/sha256.rkt`

Code extraction strategy:
- One spec or proof pattern per example
- Include relevant assertions, verify/synthesize calls, and symbolic definitions

Task: For each spec:
Analyze the code: Identify intended property and Rosette constructs used.

Generate a "User Prompt": A realistic request for verification/synthesis.

Generate a "Think" block: Mention:
- Property being verified
- Symbolic variables and constraints
- Expected solver outcomes

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Rosette Maintainer AI, an expert in verification specs."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 4–6 examples.
