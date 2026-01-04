#!/usr/bin/env python3
"""Interactive repository selector and cloner."""

import subprocess
import sys
from pathlib import Path

try:
    # Python 3.11+ has tomllib in standard library
    import tomllib as tomli
except ImportError:
    # Fallback for older Python versions
    try:
        import tomli
    except ImportError:
        print("Error: tomli library is required. Install it with: pip install tomli", file=sys.stderr)
        sys.exit(1)


def load_config():
    """Load repository configuration from config.toml."""
    config_path = Path("/app/datagen/config/config.toml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    
    return config.get("repositories", {})


def display_repositories(repos):
    """Display available repositories."""
    print("\nAvailable repositories:")
    print("-" * 40)
    repo_list = list(repos.items())
    for i, (name, url) in enumerate(repo_list, 1):
        print(f"{i}. {name}: {url}")
    print("-" * 40)


def get_user_selection(repos):
    """Prompt user to select a repository."""
    while True:
        try:
            choice = input("\nSelect a repository (enter number): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(repos):
                repo_name, repo_url = list(repos.items())[index]
                return repo_name, repo_url
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(repos)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)


def clone_repository(repo_name, repo_url, workspace_dir):
    """Clone the selected repository to the workspace directory."""
    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    repo_path = workspace_path / repo_name
    
    if repo_path.exists():
        print(f"\nRepository {repo_name} already exists at {repo_path}")
        response = input("Do you want to remove it and clone again? (y/n): ").strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(repo_path)
        else:
            print("Keeping existing repository.")
            return repo_path
    
    print(f"\nCloning {repo_name} from {repo_url}...")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(repo_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Successfully cloned {repo_name} to {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr.decode(), file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git not found. Please ensure git is installed.", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to run the repository selector."""
    repos = load_config()
    
    if not repos:
        print("No repositories configured.", file=sys.stderr)
        sys.exit(1)
    
    display_repositories(repos)
    repo_name, repo_url = get_user_selection(repos)
    
    workspace_dir = "/app/workspace"
    repo_path = clone_repository(repo_name, repo_url, workspace_dir)
    
    print(f"\nRepository {repo_name} is ready at {workspace_dir}/{repo_name}")
    
    # Process prompts after repository is cloned
    print("\n" + "="*60, file=sys.stderr)
    print("Starting prompt processing...", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)
    
    try:
        from datagen.prompt_processor import process_all_prompts
        import asyncio
        import os
        
        prompts_dir = Path("/app/datagen/config/prompts")
        output_dir = Path("/app/output")
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. Prompt processing will fail.", file=sys.stderr)
            return
        
        # Set up CODEX_HOME
        codex_home = Path("/tmp/codex_home")
        codex_home.mkdir(parents=True, exist_ok=True)
        
        # Login to codex first (required for API key authentication)
        print("Setting up codex authentication...", file=sys.stderr)
        from datagen.prompt_processor import login_codex
        login_success = login_codex(codex_home, api_key)
        if not login_success:
            print("Warning: Failed to create auth.json, but continuing anyway...", file=sys.stderr)
        
        # Process all prompts
        results = asyncio.run(
            process_all_prompts(prompts_dir, repo_path, output_dir, api_key, codex_home, max_concurrent=5)
        )
        
        # Print summary
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Prompt processing complete: {successful} succeeded, {failed} failed", file=sys.stderr)
        
        # Extract JSONL files to output directory
        if successful > 0:
            print("\nExtracting JSONL files...", file=sys.stderr)
            try:
                from datagen.extract_jsonl import extract_jsonl_files
                output_dir = Path("/app/output")
                extract_jsonl_files(repo_path, output_dir, repo_name)
                print("\nJSONL files are available in /app/output (mounted to ./datagen/output on host)", file=sys.stderr)
            except Exception as e:
                print(f"Error extracting JSONL files: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error during prompt processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Don't exit - allow user to see the error and continue if needed


if __name__ == "__main__":
    main()

