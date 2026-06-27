"""
Convert all wrapped function docstrings from method-style to function-style for both
Graph and Framework (or other @copy_doc-wrapped method classes in the future).

Dry-run by default — shows a unified diff per changed function.
Pass --apply to write changes to disk.
"""

import ast
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyrigi._utils._graph_alias_mapping import GRAPH_ALIAS_METHOD_TO_NX  # noqa: E402
from migration_targets import CLASS_TO_TARGET_FILES  # noqa: E402


def _collect_wrapper_methods(class_name: str, rel_file: str) -> set[str]:
    """
    Return names of methods in the given class decorated with @copy_doc(...).

    The class has to be in `rel_file`.
    """
    file_path = ROOT / rel_file
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    wrapper_methods = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for class_node in node.body:
            if not isinstance(class_node, ast.FunctionDef):
                continue
            if class_node.name.startswith("_"):
                continue

            is_wrapper = any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "copy_doc"
                for decorator in class_node.decorator_list
            )
            if is_wrapper:
                wrapper_methods.add(class_node.name)
    return wrapper_methods


# Collect Graph and Framework wrapper sets up front
WRAPPER_METHODS = {
    "Graph": _collect_wrapper_methods("Graph", "pyrigi/graph/graph.py"),
    "Framework": _collect_wrapper_methods("Framework", "pyrigi/framework/framework.py"),
}

# For backwards compatibility
GRAPH_WRAPPER_METHODS = WRAPPER_METHODS["Graph"]


def _transform_doctest_method_to_func(line: str, class_methods: set[str]) -> str:
    """
    Transform a single doctest line from method-style to function-style.

    ``>>> G.is_rigid()``           -> ``>>> is_rigid(G)``
    ``>>> G.is_rigid(dim=2)``      -> ``>>> is_rigid(G, dim=2)``
    ``>>> H.is_linked(1,7)``       -> ``>>> is_linked(H, 1,7)``
    ``>>> H = G.zero_extension(...)`` -> ``>>> H = zero_extension(G, ...)``
    ``>>> rigid_dim = G.max_rigid_dimension(); rigid_dim``
        -> ``>>> rigid_dim = max_rigid_dimension(G); rigid_dim``
    ``>>> len(list(G.all_k_extensions(0)))``
        -> ``>>> len(list(all_k_extensions(G, 0)))``
    ``>>> type(G.all_extensions())``
        -> ``>>> type(all_extensions(G))``

    Does NOT transform:
    - ``>>> G.add_edge(0,2)``  — add_edge is NOT in class_methods
    - ``>>> print(G)``         — print is NOT in class_methods
    - ``>>> graphs.Complete(5)``  — Complete is NOT in class_methods
    """
    m = re.match(
        r"^(\s*>>> )"  # group 1: prefix with >>>
        r"((?:\w+\s*=\s*)?)"  # group 2: optional assignment like "H = " or "rigid_dim = "
        r"(\w+)"  # group 3: variable name (G, H, etc.)
        r"\."  # the dot
        r"(\w+)"  # group 4: method name
        r"\(([^)]*)\)"  # group 5: arguments (everything inside parens)
        r"(.*)"  # group 6: suffix (e.g., "; rigid_dim" or trailing comment)
        r"$",
        line,
    )
    if m:
        prefix, assignment, var, method, args, suffix = m.groups()
        if method in class_methods:
            if args:
                new_args = f"{var}, {args}"
            else:
                new_args = var
            return f"{prefix}{assignment}{method}({new_args}){suffix}"

    # Fallback: re.sub for wrapped calls
    if not re.search(r"(?:>>>|\.\.\.)\s", line):
        return line

    def _replace(m):
        var, method, args = m.group(1), m.group(2), m.group(3)
        if method not in class_methods:
            return m.group(0)
        return f"{method}({var}, {args})" if args else f"{method}({var})"

    return re.sub(r"(\w+)\.(\w+)\(([^)]*)\)", _replace, line)


def _transform_meth_to_func_refs(line: str, class_methods: set[str]) -> str:
    """
    Transform :meth: references to :func: for bare method names.

    ``:meth:`.is_rigid```        -> ``:func:`.is_rigid```
    ``:meth:`~.is_rigid```       -> ``:func:`~.is_rigid```

    Does NOT transform (has class prefix — these stay as :meth:):
    ``:meth:`.Framework.is_inf_rigid```
    ``:meth:`~.Graph.is_k_vertex_apex```
    ``:meth:`~pyrigi.graph.Graph.is_stable_set```
    ``:meth:`~Graph.layout```
    """

    def _replace(m):
        prefix = m.group(1)
        name = m.group(2)
        if name in class_methods:
            return f":func:`{prefix}{name}`"
        return m.group(0)

    return re.sub(
        r":meth:`([~]?\.?)([A-Za-z_]\w*)(?:\(\))?`",
        _replace,
        line,
    )


def method_to_func_doc(docstring: str, class_methods: set[str]) -> str:
    """
    Convert a method-style docstring to function-style.

    Transformations applied:
    1. Doctest: ``>>> G.method(args)`` -> ``>>> method(G, args)``
    2. ``:meth:`.name``` -> ``:func:`.name``` (bare names only)
    3. ``this method`` -> ``this function``
    4. ``the method`` -> ``the function``
    5. ``The method`` -> ``The function``
    """
    if docstring is None:
        return None

    lines = docstring.split("\n")
    result = []

    for line in lines:
        # 1. Doctest lines
        new_line = _transform_doctest_method_to_func(line, class_methods)
        if new_line != line:
            result.append(new_line)
            continue

        # 2. :meth: -> :func: for bare names
        line = _transform_meth_to_func_refs(line, class_methods)

        # 3. "this method" / "the method" / "The method"
        line = re.sub(r"\bthis method\b", "this function", line)
        line = re.sub(r"\bthe method\b", "the function", line)
        line = re.sub(r"\bThe method\b", "The function", line)

        result.append(line)

    return "\n".join(result)


def _transform_alias_doctest_line(line: str) -> str:
    """Convert Graph-alias method calls in a doctest line to NetworkX forms."""
    match = re.match(
        r"^(\s*(?:>>>|\.\.\.)\s+)"
        r"((?:\w+\s*=\s*)?)"
        r"(\w+)"
        r"\."
        r"(\w+)"
        r"\(([^)]*)\)"
        r"(.*)$",
        line,
    )
    if not match:
        return line

    prefix, assignment, var, method, args, suffix = match.groups()

    if method == "is_isomorphic" and var not in ("nx", "networkx"):
        nx_args = f"{var}, {args}" if args else var
        return f"{prefix}{assignment}nx.is_isomorphic({nx_args}){suffix}"

    nx_method = GRAPH_ALIAS_METHOD_TO_NX.get(method)
    if nx_method is None:
        return line

    return f"{prefix}{assignment}{var}.{nx_method}({args}){suffix}"


def _transform_graph_alias_to_nx_doc(docstring: str) -> str:
    """Convert Graph-alias calls to NetworkX forms across all doctest lines."""
    return "\n".join(
        _transform_alias_doctest_line(line) for line in docstring.split("\n")
    )


def _quote_style(literal_text: str) -> str | None:
    """Return the triple-quote style used, or None for single-quoted/f-strings."""
    text = literal_text.lstrip("r")
    if text.startswith('"""'):
        return '"""'
    if text.startswith("'''"):
        return "'''"
    return None


def transform_file(
    filepath: Path, dry_run: bool = True, class_name: str = "Graph"
) -> int:
    """Transform all function docstrings in filepath.

    Returns number of changed functions.
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)

    replacements = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        const_node = first.value
        if not isinstance(const_node, ast.Constant) or not isinstance(
            const_node.value, str
        ):
            continue

        old_value = const_node.value
        new_value = method_to_func_doc(old_value, WRAPPER_METHODS[class_name])
        if class_name == "Graph":
            new_value = _transform_graph_alias_to_nx_doc(new_value)
        if new_value == old_value:
            continue

        orig_literal = ast.get_source_segment(source, const_node)
        if orig_literal is None:
            print(
                "  WARNING: could not get source segment for "
                f"{node.name} in {filepath.name}"
            )
            continue

        quote = _quote_style(orig_literal)
        if quote is None:
            continue  # single-quoted or f-string — skip

        raw_prefix = "r" if orig_literal.startswith("r") else ""
        new_literal = f"{raw_prefix}{quote}{new_value}{quote}"
        replacements.append((node.name, orig_literal, new_literal))

    if not replacements:
        print(f"  (no changes) {filepath.relative_to(ROOT)}")
        return 0

    print(
        f"\n=== {filepath.relative_to(ROOT)}" f" — {len(replacements)} function(s) ==="
    )

    new_source = source
    for func_name, orig_literal, new_literal in replacements:
        if dry_run:
            diff = list(
                difflib.unified_diff(
                    orig_literal.splitlines(keepends=True),
                    new_literal.splitlines(keepends=True),
                    fromfile=f"{func_name} (before)",
                    tofile=f"{func_name} (after)",
                    lineterm="",
                )
            )
            # Cap per-function diff at 60 lines to keep output readable
            print("\n".join(diff[:60]))
            if len(diff) > 60:
                print(f"  ... ({len(diff) - 60} more lines)")
        else:
            new_source = new_source.replace(orig_literal, new_literal, 1)

    if not dry_run:
        filepath.write_text(new_source, encoding="utf-8")
        print("  Written.")

    return len(replacements)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate docstrings for methods decorated with @copy_doc "
        "in the specified class."
    )
    parser.add_argument("class_name", help="Class to migrate (e.g. Graph, Framework)")
    parser.add_argument(
        "--apply", action="store_true", help="Write changes to disk (default: dry run)"
    )
    args = parser.parse_args()
    class_name = args.class_name
    dry_run = not args.apply

    print(f"\nDocstring migration for class: {class_name}\n")
    if dry_run:
        print("DRY RUN — pass --apply to write changes\n")
    else:
        print("APPLYING changes...\n")

    if class_name not in WRAPPER_METHODS:
        print(
            f"ERROR: Unknown class '{class_name}'. "
            f"Available: {list(WRAPPER_METHODS.keys())}"
        )
        sys.exit(1)

    total = 0
    target_files = CLASS_TO_TARGET_FILES[class_name]
    for filepath in target_files:
        if not filepath.exists():
            print(f"  MISSING: {filepath}")
            continue
        total += transform_file(filepath, dry_run=dry_run, class_name=class_name)

    mode = "would change" if dry_run else "changed"
    print(f"\nTotal functions {mode}: {total}")
