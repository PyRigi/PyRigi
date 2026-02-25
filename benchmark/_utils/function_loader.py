import importlib.util
import inspect
import sys
from typing import Callable, Tuple
import networkx as nx


def load_function_and_detect_param(target_str: str) -> Tuple[Callable, str]:
    """
    Load a function from a file path and detect the graph parameter name.

    Args:
        target_str: Format "path/to/file.py:function_name"

    Returns:
        Tuple of (function_object, graph_parameter_name)
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
    Compute SHA256 hash of the function's AST representation.
    This ignores superficial changes like comments and docstrings.
    This is used to detect staleness in benchmark results.
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
