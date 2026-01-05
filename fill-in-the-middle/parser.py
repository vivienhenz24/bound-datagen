#!/usr/bin/env python3
"""Parse Rust source files using tree-sitter to extract maskable AST nodes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tree_sitter_rust as ts_rust
from tree_sitter import Language, Parser, Node

from config import MASKABLE_NODE_TYPES, MIN_SPAN_LENGTH, MAX_SPAN_LENGTH


@dataclass
class ASTNode:
    """Represents a maskable AST node."""

    node_type: str
    text: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    file_path: str


def create_parser() -> Parser:
    """Create a tree-sitter parser for Rust."""
    parser = Parser(Language(ts_rust.language()))
    return parser


def find_maskable_nodes(node: Node, source_bytes: bytes, file_path: str) -> Iterator[ASTNode]:
    """
    Recursively find all maskable nodes in the AST.

    Args:
        node: The current tree-sitter node
        source_bytes: The source file as bytes
        file_path: Path to the source file

    Yields:
        ASTNode objects for each maskable node found
    """
    # Check if this node type is maskable
    if node.type in MASKABLE_NODE_TYPES:
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        # Filter by span length
        if MIN_SPAN_LENGTH <= len(text) <= MAX_SPAN_LENGTH:
            yield ASTNode(
                node_type=node.type,
                text=text,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0] + 1,  # 1-indexed
                end_line=node.end_point[0] + 1,
                file_path=file_path,
            )

    # Recursively process children
    for child in node.children:
        yield from find_maskable_nodes(child, source_bytes, file_path)


def parse_file(file_path: Path, parser: Parser) -> list[ASTNode]:
    """
    Parse a single Rust file and extract maskable nodes.

    Args:
        file_path: Path to the Rust file
        parser: The tree-sitter parser

    Returns:
        List of ASTNode objects
    """
    try:
        source_bytes = file_path.read_bytes()
    except (IOError, OSError) as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return []

    tree = parser.parse(source_bytes)
    nodes = list(find_maskable_nodes(tree.root_node, source_bytes, str(file_path)))

    return nodes


def parse_directory(directory: Path) -> list[ASTNode]:
    """
    Parse all Rust files in a directory and extract maskable nodes.

    Args:
        directory: Path to the directory containing Rust files

    Returns:
        List of all ASTNode objects found
    """
    parser = create_parser()
    all_nodes: list[ASTNode] = []

    rust_files = list(directory.rglob("*.rs"))
    print(f"Found {len(rust_files)} Rust files to parse")

    for file_path in rust_files:
        nodes = parse_file(file_path, parser)
        all_nodes.extend(nodes)

    return all_nodes


def get_node_stats(nodes: list[ASTNode]) -> dict[str, int]:
    """Get statistics about node types found."""
    stats: dict[str, int] = {}
    for node in nodes:
        stats[node.node_type] = stats.get(node.node_type, 0) + 1
    return stats


def main() -> None:
    """Main entry point for testing the parser."""
    from config import REPO_DIR

    if not REPO_DIR.exists():
        print(f"Repository not found at {REPO_DIR}")
        print("Run clone_repo.py first.")
        return

    print("Parsing Rust files...")
    nodes = parse_directory(REPO_DIR)

    print(f"\nTotal maskable nodes found: {len(nodes)}")
    print("\nNode type breakdown:")
    stats = get_node_stats(nodes)
    for node_type, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {node_type}: {count}")

    if nodes:
        print("\nSample node:")
        sample = nodes[0]
        print(f"  Type: {sample.node_type}")
        print(f"  File: {sample.file_path}")
        print(f"  Lines: {sample.start_line}-{sample.end_line}")
        print(f"  Text preview: {sample.text[:100]}...")


if __name__ == "__main__":
    main()

