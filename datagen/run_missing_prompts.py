#!/usr/bin/env python3
"""Run just the 5 missing prompts."""

import asyncio
import os
import sys
from pathlib import Path

# Import the prompt processor functions
from datagen.prompt_processor import process_all_prompts, login_codex

# The 3 remaining missing prompts
MISSING_PROMPTS = [
    "otel_setup",
    "token_extraction",
    "trace_propagation",
]


def clone_repo(repo_name: str, repo_url: str, workspace_dir: Path) -> Path:
    """Clone repository if it doesn't exist."""
    repo_path = workspace_dir / repo_name
    
    if repo_path.exists():
        print(f"Repository {repo_name} already exists at {repo_path}", file=sys.stderr)
        return repo_path
    
    print(f"Cloning {repo_name} from {repo_url}...", file=sys.stderr)
    import subprocess
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(repo_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Successfully cloned {repo_name} to {repo_path}", file=sys.stderr)
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git not found. Please ensure git is installed.", file=sys.stderr)
        sys.exit(1)


async def main():
    """Run the 5 missing prompts."""
    # Paths
    workspace_dir = Path("/app/workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    repo_path = workspace_dir / "svix"
    prompts_dir = Path("/app/datagen/config/prompts")
    output_dir = Path("/app/output")
    codex_home = Path("/tmp/codex_home")
    
    # Clone repo if it doesn't exist
    if not repo_path.exists():
        print("Repository not found. Cloning...", file=sys.stderr)
        # Default to svix repo
        clone_repo("svix", "https://github.com/svix/svix-webhooks", workspace_dir)
    
    if not prompts_dir.exists():
        print(f"Error: Prompts directory does not exist: {prompts_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    codex_home.mkdir(parents=True, exist_ok=True)
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)
    
    # Login to codex
    print("Setting up codex authentication...", file=sys.stderr)
    login_success = login_codex(codex_home, api_key)
    if not login_success:
        print("Warning: Failed to create auth.json, but continuing anyway...", file=sys.stderr)
    
    # Process only the missing prompts
    print(f"\nProcessing {len(MISSING_PROMPTS)} missing prompts...", file=sys.stderr)
    print(f"Prompts: {', '.join(MISSING_PROMPTS)}\n", file=sys.stderr)
    
    results = await process_all_prompts(
        prompts_dir=prompts_dir,
        repo_path=repo_path,
        output_dir=output_dir,
        api_key=api_key,
        codex_home=codex_home,
        max_concurrent=5,
        include_prompts=MISSING_PROMPTS,
    )
    
    # Print summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Summary: {successful} succeeded, {failed} failed out of {len(results)} total", file=sys.stderr)
    
    if failed > 0:
        print("\nFailed prompts:", file=sys.stderr)
        for prompt_name, success, error in results:
            if not success:
                print(f"  - {prompt_name}: {error[:200]}", file=sys.stderr)
    
    # Extract JSONL files if any succeeded
    if successful > 0:
        print("\nExtracting JSONL files...", file=sys.stderr)
        try:
            from datagen.extract_jsonl import extract_jsonl_files
            extract_jsonl_files(repo_path, output_dir, "svix")
            print(f"Extracted JSONL files to {output_dir}", file=sys.stderr)
            print("Files are available in /app/output (mounted to ./datagen/output on host)", file=sys.stderr)
        except Exception as e:
            print(f"Error extracting JSONL files: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # Exit with error code if any failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())

