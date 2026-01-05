#!/usr/bin/env python3
"""Evaluate base model accuracy on the FIM test set using Ollama."""

import json
import re
from pathlib import Path

import ollama
import sacrebleu
from tqdm import tqdm

from config import (
    TEST_FILE,
    MODEL_NAME,
    MAX_NEW_TOKENS,
)
from dataset import load_from_jsonl


def ensure_model_available(model_name: str) -> None:
    """
    Ensure the model is available in Ollama, pulling if necessary.

    Args:
        model_name: Ollama model name
    """
    print(f"Checking if model '{model_name}' is available...")
    try:
        ollama.show(model_name)
        print(f"Model '{model_name}' is available.")
    except ollama.ResponseError:
        print(f"Model '{model_name}' not found. Pulling...")
        ollama.pull(model_name)
        print(f"Model '{model_name}' pulled successfully.")


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize internal whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """
    Compute exact match accuracy after normalization.

    Args:
        predictions: List of predicted completions
        references: List of reference completions

    Returns:
        Exact match accuracy (0-1)
    """
    matches = 0
    for pred, ref in zip(predictions, references):
        if normalize_text(pred) == normalize_text(ref):
            matches += 1
    return matches / len(predictions) if predictions else 0.0


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Compute BLEU-4 score.

    Args:
        predictions: List of predicted completions
        references: List of reference completions

    Returns:
        BLEU-4 score
    """
    # sacrebleu expects references as list of lists
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
    return bleu.score


def generate_completion(
    model_name: str,
    prompt: str,
    max_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Generate a completion for a single FIM prompt using Ollama.

    Args:
        model_name: Ollama model name
        prompt: FIM prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Generated completion
    """
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        raw=True,  # Send prompt as-is without chat template wrapping
        options={
            "num_predict": max_tokens,
            "temperature": 0,  # Greedy decoding for reproducibility
            "stop": ["<|endoftext|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"],
        },
    )

    completion = response["response"]
    return completion


def evaluate_model(
    model_name: str,
    test_samples: list[dict],
) -> dict:
    """
    Evaluate the model on test samples.

    Args:
        model_name: Ollama model name
        test_samples: List of test samples

    Returns:
        Dictionary with evaluation metrics
    """
    prompts = [s["prompt"] for s in test_samples]
    references = [s["completion"] for s in test_samples]

    print(f"Generating completions for {len(prompts)} test samples...")

    predictions = []
    for prompt in tqdm(prompts, desc="Generating"):
        completion = generate_completion(model_name, prompt)
        predictions.append(completion)

    # Compute metrics
    exact_match = compute_exact_match(predictions, references)
    bleu_score = compute_bleu(predictions, references)

    return {
        "exact_match": exact_match,
        "bleu_score": bleu_score,
        "num_samples": len(test_samples),
        "predictions": predictions,
        "references": references,
    }


def print_examples(
    predictions: list[str],
    references: list[str],
    samples: list[dict],
    n: int = 5,
) -> None:
    """Print example predictions vs references."""
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)

    for i in range(min(n, len(predictions))):
        print(f"\n--- Example {i+1} ({samples[i]['node_type']}) ---")
        print(f"File: {samples[i]['file']}")
        print(f"Lines: {samples[i]['line_range']}")
        print(f"\nExpected:\n{references[i][:300]}{'...' if len(references[i]) > 300 else ''}")
        print(f"\nPredicted:\n{predictions[i][:300]}{'...' if len(predictions[i]) > 300 else ''}")
        match = normalize_text(predictions[i]) == normalize_text(references[i])
        print(f"\nExact match: {'✓' if match else '✗'}")


def main() -> None:
    """Main entry point for evaluation."""
    if not TEST_FILE.exists():
        print(f"Test file not found at {TEST_FILE}")
        print("Run dataset.py first to generate the dataset.")
        return

    # Load test samples
    print("Loading test samples...")
    test_samples = load_from_jsonl(TEST_FILE)
    print(f"Loaded {len(test_samples)} test samples")

    # Ensure model is available
    ensure_model_available(MODEL_NAME)

    # Evaluate
    results = evaluate_model(MODEL_NAME, test_samples)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Test samples: {results['num_samples']}")
    print(f"Exact Match Accuracy: {results['exact_match']:.2%}")
    print(f"BLEU-4 Score: {results['bleu_score']:.2f}")

    # Print examples
    print_examples(
        results["predictions"],
        results["references"],
        test_samples,
        n=5,
    )

    # Save results
    results_file = TEST_FILE.parent / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "exact_match": results["exact_match"],
                "bleu_score": results["bleu_score"],
                "num_samples": results["num_samples"],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
