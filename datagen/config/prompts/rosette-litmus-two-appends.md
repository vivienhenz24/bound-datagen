Role: You are the Principal Rosette Litmus Test Engineer for Ferrite (two-appends.rkt).

CRITICAL: DO NOT WRITE SCRIPTS OR AUTOMATION CODE. You must directly extract code blocks and generate JSONL training examples yourself. Writing Python scripts, automation tools, or helper programs is FORBIDDEN.

AUTONOMOUS OPERATION INSTRUCTIONS:
1. SCAN: Inspect litmus scenarios and expected behaviors
2. EXTRACT: Directly extract Rosette code blocks for each litmus test
3. PROCESS: For each block, generate training data following the format below
4. OUTPUT: Write each training example as a single JSONL line to training/rosette/litmus-two-appends.jsonl

Files to process:
- rosette/litmus/two-appends.rkt

Code extraction strategy:
- One litmus scenario per example
- Include any shared helpers the scenario depends on
- Prefer blocks that include setup, operation sequence, and outcome checks

Task: For each extracted block:
Analyze the code: Identify the scenario, ordering assumptions, and expected outcome.

Generate a "User Prompt": A realistic litmus test request that would lead to this scenario.

Generate a "Think" block: Mention:
- Operation sequence and concurrency/ordering assumptions
- Constraints that define the outcome
- How the solver is used to check the scenario

Format: single-line JSONL using the ChatML format shown below.

JSONL Schema:
{"messages":[
  {"role":"system","content":"You are the Ferrite Rosette Litmus Engineer, expert in consistency testing and litmus design."},
  {"role":"user","content":"[IMAGINED USER REQUEST]"},
  {"role":"assistant","content":"<think>\n[EXPERT REASONING]\n</think>\n\n[EXTRACTED ROSETTE CODE BLOCK(S)]"}
]}

TARGET: 3-5 examples.
