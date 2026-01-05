#!/usr/bin/env python3
"""Generate FIM samples from parsed AST nodes."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator
import json

from parser import ASTNode, parse_directory
from config import (
    FIM_PREFIX,
    FIM_SUFFIX,
    FIM_MIDDLE,
    MAX_CONTEXT_LENGTH,
    REPO_DIR,
)


@dataclass
class FIMSample:
    """A fill-in-the-middle sample."""

    prompt: str
    completion: str
    file: str
    node_type: str
    line_range: tuple[int, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "file": self.file,
            "node_type": self.node_type,
            "line_range": list(self.line_range),
        }


def generate_fim_sample(node: ASTNode, source_content: str) -> FIMSample | None:
    """
    Generate a FIM sample from an AST node.

    Args:
        node: The AST node to mask
        source_content: The full source file content

    Returns:
        A FIMSample or None if the sample would be invalid
    """
    # Extract prefix and suffix
    prefix = source_content[: node.start_byte]
    suffix = source_content[node.end_byte :]
    completion = node.text

    # Check context length
    if len(prefix) + len(suffix) > MAX_CONTEXT_LENGTH:
        # Truncate prefix and suffix to fit within context window
        # Keep more context around the masked region
        available = MAX_CONTEXT_LENGTH
        half = available // 2

        if len(prefix) > half and len(suffix) > half:
            prefix = prefix[-half:]
            suffix = suffix[:half]
        elif len(prefix) > half:
            prefix = prefix[-(available - len(suffix)) :]
        else:
            suffix = suffix[: available - len(prefix)]

    # Build the FIM prompt
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

    return FIMSample(
        prompt=prompt,
        completion=completion,
        file=node.file_path,
        node_type=node.node_type,
        line_range=(node.start_line, node.end_line),
    )


def generate_samples_from_file(file_path: Path, nodes: list[ASTNode]) -> Iterator[FIMSample]:
    """
    Generate FIM samples from all nodes in a file.

    Args:
        file_path: Path to the source file
        nodes: List of AST nodes from this file

    Yields:
        FIMSample objects
    """
    if not nodes:
        return

    try:
        source_content = file_path.read_text(encoding="utf-8", errors="replace")
    except (IOError, OSError) as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return

    for node in nodes:
        sample = generate_fim_sample(node, source_content)
        if sample is not None:
            yield sample


def generate_all_samples(nodes: list[ASTNode]) -> list[FIMSample]:
    """
    Generate FIM samples from all parsed nodes.

    Args:
        nodes: List of all AST nodes

    Returns:
        List of FIMSample objects
    """
    # Group nodes by file
    nodes_by_file: dict[str, list[ASTNode]] = {}
    for node in nodes:
        if node.file_path not in nodes_by_file:
            nodes_by_file[node.file_path] = []
        nodes_by_file[node.file_path].append(node)

    samples: list[FIMSample] = []
    for file_path, file_nodes in nodes_by_file.items():
        file_samples = list(generate_samples_from_file(Path(file_path), file_nodes))
        samples.extend(file_samples)

    return samples


def deduplicate_samples(samples: list[FIMSample]) -> list[FIMSample]:
    """
    Remove duplicate samples based on completion text.

    Args:
        samples: List of FIM samples

    Returns:
        Deduplicated list of samples
    """
    seen_completions: set[str] = set()
    unique_samples: list[FIMSample] = []

    for sample in samples:
        # Normalize whitespace for comparison
        normalized = " ".join(sample.completion.split())
        if normalized not in seen_completions:
            seen_completions.add(normalized)
            unique_samples.append(sample)

    return unique_samples


def main() -> None:
    """Main entry point for testing the generator."""
    if not REPO_DIR.exists():
        print(f"Repository not found at {REPO_DIR}")
        print("Run clone_repo.py first.")
        return

    print("Parsing Rust files...")
    nodes = parse_directory(REPO_DIR)
    print(f"Found {len(nodes)} maskable nodes")

    print("\nGenerating FIM samples...")
    samples = generate_all_samples(nodes)
    print(f"Generated {len(samples)} samples")

    print("\nDeduplicating samples...")
    samples = deduplicate_samples(samples)
    print(f"After deduplication: {len(samples)} samples")

    if samples:
        print("\nSample FIM entry:")
        sample = samples[0]
        print(f"  File: {sample.file}")
        print(f"  Node type: {sample.node_type}")
        print(f"  Lines: {sample.line_range}")
        print(f"  Completion: {sample.completion[:100]}...")
        print(f"  Prompt length: {len(sample.prompt)} chars")


if __name__ == "__main__":
    main()

