"""
convert all wrapped function docstrings
from method-style to function-style.

Dry-run by default — shows a unified diff per changed function.
Pass --apply to write changes to disk.
"""

import ast
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrigi._utils._doc import method_to_func_doc  # noqa: E402
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


def _quote_style(literal_text: str) -> str | None:
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
