#!/usr/bin/env python3
"""Clone the svix-webhooks repository."""

import subprocess
import sys
from pathlib import Path

from config import DATA_DIR, REPO_DIR, REPO_URL


def clone_repo() -> None:
    """Clone the svix-webhooks repository if it doesn't exist."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if REPO_DIR.exists():
        print(f"Repository already exists at {REPO_DIR}")
        print("Pulling latest changes...")
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=REPO_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            print("Repository updated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to pull latest changes: {e.stderr}")
        return

    print(f"Cloning {REPO_URL} to {REPO_DIR}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    clone_repo()

    # Print some stats about the cloned repo
    rust_files = list(REPO_DIR.rglob("*.rs"))
    print(f"\nFound {len(rust_files)} Rust files in the repository.")


if __name__ == "__main__":
    main()

