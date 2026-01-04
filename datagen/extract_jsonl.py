#!/usr/bin/env python3
"""Extract JSONL files from workspace to output directory."""

import shutil
import sys
from pathlib import Path


def extract_jsonl_files(repo_path: Path, output_dir: Path, repo_name: str):
    """
    Extract JSONL files from repository workspace to output directory.
    
    Args:
        repo_path: Path to the cloned repository (where JSONL files were generated)
        output_dir: Directory where JSONL files should be copied
        repo_name: Name of the repository (for prefixing files)
    """
    # Find all JSONL files in the repository
    jsonl_files = list(repo_path.rglob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {repo_path}", file=sys.stderr)
        return
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFound {len(jsonl_files)} JSONL file(s) to extract", file=sys.stderr)
    
    extracted_count = 0
    for jsonl_file in jsonl_files:
        # Create destination path with repo name prefix to avoid conflicts
        # Use the filename from the prompt (e.g., newtype_wrappers.jsonl)
        dest_name = f"{repo_name}_{jsonl_file.name}" if repo_name else jsonl_file.name
        dest_path = output_dir / dest_name
        
        try:
            shutil.copy2(jsonl_file, dest_path)
            print(f"  Extracted: {jsonl_file.name} -> {dest_path.name}", file=sys.stderr)
            extracted_count += 1
        except Exception as e:
            print(f"  Error copying {jsonl_file.name}: {e}", file=sys.stderr)
    
    print(f"\nExtracted {extracted_count} JSONL file(s) to {output_dir}", file=sys.stderr)


def main():
    """Main function to extract JSONL files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract JSONL files from workspace")
    parser.add_argument("--repo-path", required=True, help="Path to cloned repository")
    parser.add_argument("--output-dir", default="/app/output", help="Output directory")
    parser.add_argument("--repo-name", help="Repository name (for file prefixing)")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path)
    output_dir = Path(args.output_dir)
    
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
        sys.exit(1)
    
    # Extract repo name from path if not provided
    repo_name = args.repo_name or repo_path.name
    
    extract_jsonl_files(repo_path, output_dir, repo_name)


if __name__ == "__main__":
    main()


