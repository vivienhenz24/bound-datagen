#!/usr/bin/env python3
"""Entry point for running the datagen application in Docker."""

import subprocess
import sys


def main():
    """Build and run the Docker container."""
    try:
        # Build and run using docker-compose
        subprocess.run(
            ["docker-compose", "up", "--build"],
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

