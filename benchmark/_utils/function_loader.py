"""
Dynamic loading of the function under test, and change detection for it.

load_function_and_detect_param - import a function by path and find its graph argument.
get_function_hash              - hash a function's code, ignoring comments and docstrings.

The hash is what makes stored results self-describing: each entry records the
hash of the code that produced it, so check_staleness.py can later report which
results no longer match the current implementation.
"""

import importlib.util
import inspect
import sys
from typing import Callable, Tuple
import networkx as nx


def load_function_and_detect_param(target_str: str) -> Tuple[Callable, str]:
    """
    Load a function from a file path and detect the graph parameter name.

    The graph argument is detected in three steps, stopping at the first hit:
    a parameter annotated as networkx.Graph, then a parameter whose name
    contains "graph", then the first parameter.

    Args:
        target_str: Format "path/to/file.py:function_name"

    Returns:
        Tuple of (function_object, graph_parameter_name)

    Raises:
        ValueError: If target_str is malformed, or the function takes no
            arguments so no graph parameter can be detected.
        ImportError: If the module cannot be loaded from the given path.
        AttributeError: If the module has no function of that name.
    """
    try:
        file_path, func_name = target_str.split(":")
    except ValueError:
        raise ValueError(
            f"Invalid target format '{target_str}'. Expected 'path/to/file.py:func_name'"
        )

    spec = importlib.util.spec_from_file_location("dynamic_bench_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_bench_module"] = module
    spec.loader.exec_module(module)

    try:
        func = getattr(module, func_name)
    except AttributeError:
        raise AttributeError(f"Function '{func_name}' not found in {file_path}")

    sig = inspect.signature(func)

    # Heuristic 1: Type hint is networkx.Graph
    for name, param in sig.parameters.items():
        if param.annotation == nx.Graph:
            return func, name

    # Heuristic 2: Argument name contains "graph"
    for name in sig.parameters:
        if "graph" in name.lower():
            return func, name

    # Fallback: Return first argument
    if sig.parameters:
        return func, list(sig.parameters.keys())[0]

    raise ValueError(f"Could not detect graph parameter for function '{func_name}'")


def get_function_hash(func: Callable) -> str:
    """
    Compute a SHA256 hash of the function's AST representation.

    Comments and docstrings are removed before hashing, so reformatting or
    re-documenting a function does not invalidate results measured from it.
    Only a change to the code itself changes the hash.

    Args:
        func: The function to hash.

    Returns:
        The first 16 hex characters of the digest, or the literal
        "unknown_hash" if the source could not be read or parsed. This
        degrades rather than raising so a benchmark run is never blocked by
        a function whose source is unavailable.
    """
    import ast
    import inspect
    import hashlib

    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)

        # Remove docstrings
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                if node.body and isinstance(node.body[0], ast.Expr):
                    if hasattr(node.body[0].value, "value") and isinstance(
                        node.body[0].value.value, str
                    ):
                        node.body.pop(0)

        # Removes comments
        clean_source = ast.unparse(tree)
        # Normalize line endings
        clean_source = clean_source.replace("\r\n", "\n")

        return hashlib.sha256(clean_source.encode("utf-8")).hexdigest()[:16]
    except Exception as e:
        print(f"Warning: Could not compute source hash for {func.__name__}: {e}")
        return "unknown_hash"
