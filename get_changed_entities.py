import ast
import os
import re
import subprocess
from typing import Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable


def get_changed_lines(file_path):
    abs_file_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_file_path)

    result = subprocess.run(
        ["git", "-C", file_dir, "diff", "-U0", file_path],
        stdout=subprocess.PIPE,
        text=True,
    )

    lines = result.stdout.splitlines()
    line_change_regex = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    modified_lines = []

    for line in lines:
        match = line_change_regex.match(line)
        if match:
            start_line = int(match.group(1))
            num_lines = int(match.group(2) or "1")

            # Collect all affected line numbers
            for i in range(num_lines):
                modified_lines.append(start_line + i)

    return modified_lines


class NodeVisitor(ast.NodeVisitor):
    """
    Custom AST node visitor that tracks parent-child relationships.
    """

    def __init__(self):
        self.parent_map = {}

    def visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
        super().visit(node)


@runtime_checkable
class ASTNodeWithLines(Protocol):
    lineno: int
    end_lineno: int | None


def get_node_line_range(node: ASTNodeWithLines) -> Tuple[int, int]:
    """
    Get the line range (start_line, end_line) for an AST node.

    Args:
        node: The AST node to get the line range for

    Returns:
        A tuple containing the start and end line numbers
    """
    start_line = node.lineno
    end_line = getattr(node, "end_lineno", start_line)
    return start_line, end_line


def is_node_in_lines(node: ast.AST, changed_lines: List[int]) -> bool:
    """
    Check if an AST node has any lines that were changed.

    Args:
        node: The AST node to check
        changed_lines: List of line numbers that were changed

    Returns:
        True if any line in the node was changed, False otherwise
    """
    if isinstance(node, ASTNodeWithLines):
        start_line, end_line = get_node_line_range(node)
        return any(start_line <= line <= end_line for line in changed_lines)
    return False


def get_parent_class(
    node: ast.AST, parent_map: Dict[ast.AST, ast.AST]
) -> Optional[ast.ClassDef]:
    """
    Get the parent class of a node if it exists.

    Args:
        node: The AST node to check
        parent_map: Dictionary mapping nodes to their parents

    Returns:
        The parent ClassDef node if the node is a method, None otherwise
    """
    parent = parent_map.get(node)
    if parent and isinstance(parent, ast.ClassDef):
        return parent
    return None


def get_changed_entities(file_path: str) -> Dict[str, Set[str]]:
    """
    Get a dictionary of changed entities (functions, methods, classes) in a file.

    Args:
        file_path: Path to the Python file

    Returns:
        Dictionary with keys 'functions', 'classes', and 'methods' containing sets of changed entity names
    """
    changed_lines = get_changed_lines(file_path)

    if not changed_lines:
        return {"functions": set(), "classes": set(), "methods": set()}

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    visitor = NodeVisitor()
    visitor.visit(tree)
    parent_map = visitor.parent_map

    changed_functions = set()
    changed_classes = set()
    changed_methods = set()

    classes_with_changed_methods = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_node_in_lines(node, changed_lines):
                parent_class = get_parent_class(node, parent_map)

                if parent_class:
                    method_name = f"{parent_class.name}.{node.name}"
                    changed_methods.add(method_name)
                    classes_with_changed_methods.add(parent_class.name)
                else:
                    changed_functions.add(node.name)

        elif isinstance(node, ast.ClassDef):
            if is_node_in_lines(node, changed_lines):
                changed_classes.add(node.name)

    changed_classes.update(classes_with_changed_methods)

    return {
        "functions": changed_functions,
        "classes": changed_classes,
        "methods": changed_methods,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify changed Python entities in a file"
    )
    parser.add_argument("file_path", help="Path to the Python file to analyze")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    changed_entities = get_changed_entities(args.file_path)

    if args.json:
        import json

        json_output = {
            "functions": list(changed_entities["functions"]),
            "classes": list(changed_entities["classes"]),
            "methods": list(changed_entities["methods"]),
        }
        print(json.dumps(json_output, indent=2))
    else:
        print("Changed functions:", changed_entities["functions"])
        print("Changed classes:", changed_entities["classes"])
        print("Changed methods:", changed_entities["methods"])
