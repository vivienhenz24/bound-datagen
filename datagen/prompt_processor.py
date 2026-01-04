#!/usr/bin/env python3
"""Process all prompts in parallel using codex-exec."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


async def run_codex_exec(
    prompt_path: Path,
    repo_path: Path,
    output_dir: Path,
    codex_home: Path,
) -> Tuple[str, bool, str]:
    """
    Run codex-exec for a single prompt file.
    
    Args:
        prompt_path: Path to the prompt file
        repo_path: Path to the cloned repository
        output_dir: Directory where JSONL files will be written
        api_key: OpenAI API key
        
    Returns:
        Tuple of (prompt_name, success, error_message)
    """
    prompt_name = prompt_path.stem
    
    # Read the prompt content
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
    except Exception as e:
        return (prompt_name, False, f"Failed to read prompt file: {e}")
    
    # Build codex-exec command
    cmd = [
        "codex-exec",
        "--model", "gpt-5.1-codex",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--cd", str(repo_path),
        "-",  # Read prompt from stdin
    ]
    
    # Set up environment with CODEX_HOME
    # Note: API key should already be stored in auth.json from login
    env = os.environ.copy()
    # Set CODEX_HOME to a directory in the container that exists
    # This prevents the "failed to install system skills" error
    codex_home = Path("/tmp/codex_home")
    codex_home.mkdir(parents=True, exist_ok=True)
    env["CODEX_HOME"] = str(codex_home)
    
    print(f"Processing prompt: {prompt_name}...", file=sys.stderr)
    
    try:
        # Run codex-exec with prompt piped via stdin
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(repo_path),
        )
        
        # Send prompt content to stdin
        stdout, stderr = await process.communicate(input=prompt_content.encode("utf-8"))
        
        if process.returncode == 0:
            print(f"✓ Completed: {prompt_name}", file=sys.stderr)
            return (prompt_name, True, "")
        else:
            error_msg = stderr.decode("utf-8", errors="replace")
            stdout_msg = stdout.decode("utf-8", errors="replace")
            # Combine stderr and stdout for full error context
            full_error = f"{error_msg}\n{stdout_msg}".strip()
            # Print the error, showing the END of the output where the actual error usually is
            print(f"✗ Failed: {prompt_name} (exit code: {process.returncode})", file=sys.stderr)
            if full_error:
                print(f"  Error output (showing last 2000 chars where error usually appears):", file=sys.stderr)
                # Show the END of the error, not the beginning, since that's where the actual failure is
                if len(full_error) > 2000:
                    error_preview = full_error[-2000:]
                    print(f"  ... (showing last 2000 of {len(full_error)} chars)", file=sys.stderr)
                else:
                    error_preview = full_error
                for line in error_preview.split('\n'):
                    print(f"    {line}", file=sys.stderr)
            else:
                print(f"  No error output captured", file=sys.stderr)
            return (prompt_name, False, full_error)
            
    except Exception as e:
        error_msg = f"Exception running codex-exec: {e}"
        print(f"✗ Error: {prompt_name} - {error_msg}", file=sys.stderr)
        return (prompt_name, False, error_msg)


def login_codex(codex_home: Path, api_key: str) -> bool:
    """
    Create auth.json file for codex with API key.
    
    Args:
        codex_home: Path to CODEX_HOME directory
        api_key: OpenAI API key
        
    Returns:
        True if login successful, False otherwise
    """
    try:
        import json
        from datetime import datetime, timezone
        
        # Ensure codex_home exists
        codex_home.mkdir(parents=True, exist_ok=True)
        
        # Create auth.json with the API key
        # Format matches AuthDotJson structure from codex-rs
        auth_data = {
            "OPENAI_API_KEY": api_key,
            "tokens": None,
            "last_refresh": None
        }
        
        auth_file = codex_home / "auth.json"
        with open(auth_file, "w") as f:
            json.dump(auth_data, f, indent=2)
        
        print("✓ Created auth.json for codex", file=sys.stderr)
        return True
    except Exception as e:
        print(f"✗ Error creating auth.json: {e}", file=sys.stderr)
        return False


async def process_all_prompts(
    prompts_dir: Path,
    repo_path: Path,
    output_dir: Path,
    api_key: str,
    codex_home: Path,
    max_concurrent: int = 5,
    include_prompts: List[str] = None,
) -> List[Tuple[str, bool, str]]:
    """
    Process all prompt files in parallel.
    
    Args:
        prompts_dir: Directory containing prompt files
        repo_path: Path to the cloned repository
        output_dir: Directory where JSONL files will be written
        api_key: OpenAI API key
        codex_home: Path to CODEX_HOME directory
        max_concurrent: Maximum number of concurrent executions
        include_prompts: Optional list of prompt names to include (e.g., ['otel_setup', 'relation_definitions'])
        
    Returns:
        List of (prompt_name, success, error_message) tuples
    """
    # Find all .txt prompt files
    prompt_files = sorted(prompts_dir.glob("*.txt"))
    
    # Filter to only included prompts if specified
    if include_prompts:
        # Create set of expected filenames (e.g., 'otel_setup' -> 'otel_setup_agent_prompt.txt')
        included_set = {f"{name}_agent_prompt.txt" for name in include_prompts}
        prompt_files = [f for f in prompt_files if f.name in included_set]
    
    if not prompt_files:
        print(f"No prompt files found in {prompts_dir}", file=sys.stderr)
        return []
    
    print(f"\nFound {len(prompt_files)} prompt files to process", file=sys.stderr)
    print(f"Processing with max {max_concurrent} concurrent executions...\n", file=sys.stderr)
    
    # Create semaphore to limit concurrent executions
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(prompt_path: Path):
        async with semaphore:
            return await run_codex_exec(prompt_path, repo_path, output_dir, codex_home)
    
    # Run all prompts in parallel
    tasks = [run_with_semaphore(prompt_path) for prompt_path in prompt_files]
    results = await asyncio.gather(*tasks)
    
    return results


def main():
    """Main function to process all prompts."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process prompts using codex-exec")
    parser.add_argument("--repo-path", required=True, help="Path to cloned repository")
    parser.add_argument("--prompts-dir", default="/app/datagen/config/prompts", help="Directory containing prompt files")
    parser.add_argument("--output-dir", default="/app/output", help="Directory for output JSONL files")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent executions")
    parser.add_argument("--include-prompts", help="Comma-separated list of prompt names to process (e.g., 'otel_setup,relation_definitions')")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path)
    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir)
    
    # Validate paths
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
        sys.exit(1)
    
    if not prompts_dir.exists():
        print(f"Error: Prompts directory does not exist: {prompts_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)
    
    # Set up CODEX_HOME
    codex_home = Path("/tmp/codex_home")
    codex_home.mkdir(parents=True, exist_ok=True)
    
    # Login to codex first (required for API key authentication)
    print("Setting up codex authentication...", file=sys.stderr)
    login_success = login_codex(codex_home, api_key)
    if not login_success:
        print("Warning: Failed to create auth.json, but continuing anyway...", file=sys.stderr)
    
    # Set up CODEX_HOME
    codex_home = Path("/tmp/codex_home")
    codex_home.mkdir(parents=True, exist_ok=True)
    
    # Login to codex first (required for API key authentication)
    print("Setting up codex authentication...", file=sys.stderr)
    login_success = login_codex(codex_home, api_key)
    if not login_success:
        print("Warning: Failed to create auth.json, but continuing anyway...", file=sys.stderr)
    
    # Parse included prompts if specified
    include_prompts = None
    if args.include_prompts:
        include_prompts = [name.strip() for name in args.include_prompts.split(",")]
        print(f"Filtering to {len(include_prompts)} specific prompts: {', '.join(include_prompts)}", file=sys.stderr)
    
    # Process all prompts
    results = asyncio.run(
        process_all_prompts(prompts_dir, repo_path, output_dir, api_key, codex_home, args.max_concurrent, include_prompts)
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
                print(f"  - {prompt_name}: {error[:100]}", file=sys.stderr)
    
    # Exit with error code if any failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()


