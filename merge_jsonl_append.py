#!/usr/bin/env python3
"""
Merge JSONL files by appending new data to merged_rosette_final.jsonl
This script finds all individual .jsonl files in the output directory and appends
any new ones to the merged file without overwriting existing data.
"""

import json
import os
from pathlib import Path
from typing import Set

# Configuration
OUTPUT_DIR = Path(__file__).parent / "datagen" / "output"
MERGED_FILE = OUTPUT_DIR / "merged_rosette_final.jsonl"


def get_existing_hashes(merged_file: Path) -> Set[str]:
    """
    Read the merged file and create a set of hashes to avoid duplicates.
    We'll use a simple hash of the entire JSON line for deduplication.
    """
    existing = set()

    if not merged_file.exists():
        print(f"Merged file does not exist yet: {merged_file}")
        return existing

    print(f"Reading existing entries from {merged_file}...")
    with open(merged_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                # Use the line content as hash for deduplication
                existing.add(hash(line))

    print(f"Found {len(existing)} existing entries")
    return existing


def merge_jsonl_files(output_dir: Path, merged_file: Path):
    """
    Append new JSONL files to the merged file.
    """
    # Get existing entries to avoid duplicates
    existing_hashes = get_existing_hashes(merged_file)

    # Find all individual jsonl files (exclude the merged file itself)
    jsonl_files = [
        f for f in output_dir.glob("*.jsonl")
        if f.name != merged_file.name
    ]

    if not jsonl_files:
        print("No JSONL files found to merge!")
        return

    print(f"\nFound {len(jsonl_files)} individual JSONL files")

    # Track statistics
    new_entries = 0
    duplicate_entries = 0
    error_count = 0

    # Open merged file in append mode
    with open(merged_file, 'a', encoding='utf-8') as out_f:
        for jsonl_file in sorted(jsonl_files):
            print(f"\nProcessing: {jsonl_file.name}")

            try:
                with open(jsonl_file, 'r', encoding='utf-8') as in_f:
                    file_new_entries = 0
                    file_duplicates = 0

                    for line_num, line in enumerate(in_f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Validate JSON
                            json.loads(line)

                            # Check for duplicates
                            line_hash = hash(line)
                            if line_hash in existing_hashes:
                                duplicate_entries += 1
                                file_duplicates += 1
                            else:
                                # Append to merged file
                                out_f.write(line + '\n')
                                existing_hashes.add(line_hash)
                                new_entries += 1
                                file_new_entries += 1

                        except json.JSONDecodeError as e:
                            print(f"  ⚠ Warning: Invalid JSON at line {line_num}: {e}")
                            error_count += 1

                    print(f"  ✓ Added {file_new_entries} new entries, skipped {file_duplicates} duplicates")

            except Exception as e:
                print(f"  ✗ Error reading {jsonl_file.name}: {e}")
                error_count += 1

    # Print summary
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    print(f"New entries added:      {new_entries}")
    print(f"Duplicate entries:      {duplicate_entries}")
    print(f"Errors encountered:     {error_count}")
    print(f"Total entries in merged: {len(existing_hashes)}")
    print(f"\nMerged file: {merged_file}")
    print("="*60)


def main():
    """Main entry point."""
    print("JSONL Merge Tool (Append Mode)")
    print("="*60)

    if not OUTPUT_DIR.exists():
        print(f"Error: Output directory not found: {OUTPUT_DIR}")
        return

    merge_jsonl_files(OUTPUT_DIR, MERGED_FILE)


if __name__ == "__main__":
    main()
