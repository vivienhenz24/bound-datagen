#!/usr/bin/env python3
"""Create train/test split and save dataset to JSONL files."""

import json
import random
from collections import defaultdict
from pathlib import Path

from parser import parse_directory, get_node_stats
from fim_generator import generate_all_samples, deduplicate_samples, FIMSample
from config import (
    REPO_DIR,
    DATA_DIR,
    TRAIN_FILE,
    TEST_FILE,
    TEST_RATIO,
    RANDOM_SEED,
)


def stratified_split(
    samples: list[FIMSample], test_ratio: float, seed: int
) -> tuple[list[FIMSample], list[FIMSample]]:
    """
    Split samples into train and test sets, stratified by node type.

    Args:
        samples: List of all FIM samples
        test_ratio: Fraction of samples to use for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, test_samples)
    """
    random.seed(seed)

    # Group samples by node type
    by_type: dict[str, list[FIMSample]] = defaultdict(list)
    for sample in samples:
        by_type[sample.node_type].append(sample)

    train_samples: list[FIMSample] = []
    test_samples: list[FIMSample] = []

    # Split each group
    for node_type, type_samples in by_type.items():
        random.shuffle(type_samples)
        n_test = max(1, int(len(type_samples) * test_ratio))

        test_samples.extend(type_samples[:n_test])
        train_samples.extend(type_samples[n_test:])

    # Shuffle the final lists
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    return train_samples, test_samples


def save_to_jsonl(samples: list[FIMSample], output_path: Path) -> None:
    """
    Save samples to a JSONL file.

    Args:
        samples: List of FIM samples
        output_path: Path to the output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            json.dump(sample.to_dict(), f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(samples)} samples to {output_path}")


def load_from_jsonl(input_path: Path) -> list[dict]:
    """
    Load samples from a JSONL file.

    Args:
        input_path: Path to the JSONL file

    Returns:
        List of sample dictionaries
    """
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def print_dataset_stats(train: list[FIMSample], test: list[FIMSample]) -> None:
    """Print statistics about the generated dataset."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    print(f"\nTotal samples: {len(train) + len(test)}")
    print(f"  Training: {len(train)} ({100 * len(train) / (len(train) + len(test)):.1f}%)")
    print(f"  Test: {len(test)} ({100 * len(test) / (len(train) + len(test)):.1f}%)")

    # Node type breakdown
    print("\nNode type breakdown (train / test):")
    train_counts: dict[str, int] = defaultdict(int)
    test_counts: dict[str, int] = defaultdict(int)

    for sample in train:
        train_counts[sample.node_type] += 1
    for sample in test:
        test_counts[sample.node_type] += 1

    all_types = set(train_counts.keys()) | set(test_counts.keys())
    for node_type in sorted(all_types):
        print(f"  {node_type}: {train_counts[node_type]} / {test_counts[node_type]}")

    # Completion length stats
    train_lengths = [len(s.completion) for s in train]
    test_lengths = [len(s.completion) for s in test]

    print("\nCompletion length statistics:")
    print(f"  Train - min: {min(train_lengths)}, max: {max(train_lengths)}, avg: {sum(train_lengths)/len(train_lengths):.1f}")
    print(f"  Test  - min: {min(test_lengths)}, max: {max(test_lengths)}, avg: {sum(test_lengths)/len(test_lengths):.1f}")


def main() -> None:
    """Main entry point to generate the dataset."""
    if not REPO_DIR.exists():
        print(f"Repository not found at {REPO_DIR}")
        print("Run clone_repo.py first.")
        return

    # Parse files
    print("Step 1: Parsing Rust files...")
    nodes = parse_directory(REPO_DIR)
    print(f"Found {len(nodes)} maskable nodes")

    node_stats = get_node_stats(nodes)
    print("\nNode type breakdown:")
    for node_type, count in sorted(node_stats.items(), key=lambda x: -x[1]):
        print(f"  {node_type}: {count}")

    # Generate samples
    print("\nStep 2: Generating FIM samples...")
    samples = generate_all_samples(nodes)
    print(f"Generated {len(samples)} samples")

    # Deduplicate
    print("\nStep 3: Deduplicating samples...")
    samples = deduplicate_samples(samples)
    print(f"After deduplication: {len(samples)} samples")

    if not samples:
        print("No samples generated. Check if the repository contains Rust files.")
        return

    # Split
    print(f"\nStep 4: Splitting into train/test ({1-TEST_RATIO:.0%}/{TEST_RATIO:.0%})...")
    train_samples, test_samples = stratified_split(samples, TEST_RATIO, RANDOM_SEED)

    # Save
    print("\nStep 5: Saving to JSONL files...")
    save_to_jsonl(train_samples, TRAIN_FILE)
    save_to_jsonl(test_samples, TEST_FILE)

    # Print stats
    print_dataset_stats(train_samples, test_samples)

    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print(f"  Train: {TRAIN_FILE}")
    print(f"  Test: {TEST_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()

