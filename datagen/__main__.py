#!/usr/bin/env python3
"""Entry point for running the datagen application in Docker."""

import subprocess
import sys


def main():
    """Build and run the Docker container with interactive repository selection."""
    try:
        # Build the Docker image first
        print("Building Docker image...")
        subprocess.run(
            ["docker-compose", "build"],
            check=True
        )
        
        # Run the container interactively with the repo selector
        print("\nStarting container...")
        print("You will be prompted to select a repository to clone.\n")
        subprocess.run(
            ["docker-compose", "run", "--rm", "app", "python", "-m", "datagen.repo_selector"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running docker-compose: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: docker-compose not found. Please install Docker Compose.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

