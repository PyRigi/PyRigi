"""
Convert all wrapped function docstrings from method-style to function-style.

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
from migration_targets import TARGET_FILES  # noqa: E402


def _collect_graph_wrapper_methods() -> set[str]:
    """Return names of Graph methods decorated with @copy_doc(...)."""
    graph_path = ROOT / "pyrigi/graph/graph.py"
    graph_source = graph_path.read_text(encoding="utf-8")
    tree = ast.parse(graph_source)

    wrapper_methods = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Graph":
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


GRAPH_WRAPPER_METHODS = _collect_graph_wrapper_methods()


def _transform_doctest_method_to_func(line: str, class_methods: set[str]) -> str:
    """
    Transform a single doctest line from method-style to function-style.

    ``>>> G.is_rigid()``           -> ``>>> is_rigid(G)``
    ``>>> G.is_rigid(dim=2)``      -> ``>>> is_rigid(G, dim=2)``
    ``>>> H.is_linked(1,7)``       -> ``>>> is_linked(H, 1,7)``
    ``>>> H = G.zero_extension(...)`` -> ``>>> H = zero_extension(G, ...)``
    ``>>> rigid_dim = G.max_rigid_dimension(); rigid_dim``
        -> ``>>> rigid_dim = max_rigid_dimension(G); rigid_dim``

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

    return line


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
    if literal_text.startswith('"""'):
        return '"""'
    if literal_text.startswith("'''"):
        return "'''"
    return None


def transform_file(filepath: Path, dry_run: bool = True) -> int:
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
        new_value = method_to_func_doc(old_value, GRAPH_WRAPPER_METHODS)
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

        new_literal = f"{quote}{new_value}{quote}"
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
    dry_run = "--apply" not in sys.argv
    if dry_run:
        print("DRY RUN — pass --apply to write changes\n")
    else:
        print("APPLYING changes...\n")

    total = 0
    for filepath in TARGET_FILES:
        if not filepath.exists():
            print(f"  MISSING: {filepath}")
            continue
        total += transform_file(filepath, dry_run=dry_run)

    mode = "would change" if dry_run else "changed"
    print(f"\nTotal functions {mode}: {total}")
