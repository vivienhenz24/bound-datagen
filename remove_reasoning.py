#!/usr/bin/env python3
"""
Script to remove <think> blocks from finetune-data.jsonl
Creates a new file with reasoning blocks removed from assistant messages.
"""

import json
import re
import sys
from pathlib import Path


def remove_reasoning_block(content: str) -> str:
    """
    Remove reasoning blocks from content.
    Handles <think>...</think> blocks.
    Also removes any leading/trailing whitespace and newlines.
    """
    # Pattern to match reasoning blocks with any content inside
    # Uses non-greedy matching and handles newlines
    # Matches the tag and any whitespace/newlines immediately after the closing tag
    pattern = r'<think>.*?</think>\s*\n?\s*'
    
    # Remove the reasoning block
    cleaned = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Clean up any extra whitespace/newlines at the start only
    # (preserve trailing whitespace as it might be intentional)
    cleaned = cleaned.lstrip()
    
    return cleaned


def process_jsonl(input_path: str, output_path: str):
    """
    Process the JSONL file and remove reasoning blocks from assistant messages.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the JSON object
                data = json.loads(line)
                
                # Process messages if they exist
                if 'messages' in data:
                    for message in data['messages']:
                        if message.get('role') == 'assistant' and 'content' in message:
                            # Remove reasoning blocks from assistant content
                            message['content'] = remove_reasoning_block(message['content'])
                
                # Write the modified JSON object
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}", 
                      file=sys.stderr)
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                continue
    
    print(f"Processed {processed_count} entries.")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python remove_reasoning.py <input.jsonl> [output.jsonl]")
        print("If output.jsonl is not specified, defaults to input_no_reasoning.jsonl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Default output name: add _no_reasoning before .jsonl
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_no_reasoning{input_file.suffix}")
    
    process_jsonl(input_path, output_path)

