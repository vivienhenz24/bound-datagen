#!/usr/bin/env python3
"""Entry point for running the datagen application in Docker."""

import subprocess
import sys
from pathlib import Path


def main():
    """Build and run the Docker container with interactive repository selection."""
    try:
        # Ensure output directory exists on host
        output_dir = Path("datagen/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory ready: {output_dir.absolute()}")
        
        # Check for .env file
        env_file = Path(".env")
        if not env_file.exists():
            print("Warning: .env file not found. OPENAI_API_KEY must be set in .env", file=sys.stderr)
        else:
            print("Found .env file")
        
        # Build the Docker image first
        print("\nBuilding Docker image...")
        subprocess.run(
            ["docker-compose", "build"],
            check=True
        )
        
        # Run the container interactively with the repo selector
        print("\nStarting container...")
        print("You will be prompted to select a repository to clone.")
        print("After cloning, all prompts will be processed automatically.")
        print("Generated JSONL files will be available in ./datagen/output/\n")
        subprocess.run(
            ["docker-compose", "run", "--rm", "app", "python", "-m", "datagen.repo_selector"],
            check=True
        )
        
        # Check if any JSONL files were generated
        jsonl_files = list(output_dir.glob("*.jsonl"))
        if jsonl_files:
            print(f"\n✓ Successfully generated {len(jsonl_files)} JSONL file(s) in {output_dir.absolute()}")
        else:
            print(f"\nNo JSONL files found in {output_dir.absolute()}")
            print("This may indicate that prompt processing failed or no files were generated.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running docker-compose: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: docker-compose not found. Please install Docker Compose.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

