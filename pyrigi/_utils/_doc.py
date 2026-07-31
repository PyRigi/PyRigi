import ast
import os
import re
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from ._graph_alias_mapping import GRAPH_ALIAS_NX_TO_METHOD

P = ParamSpec("P")
T = TypeVar("T")

try:
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def skip_execution(line, cell):  # noqa: U100
        print(
            "This cell was marked to be skipped (probably due to long execution time)."
        )
        print("Remove the cell magic `%%skip_execution` to run it.")
        return

except NameError:

    def skip_execution():
        pass


def copy_doc(
    proxy_func: Callable[P, T],
) -> Callable[[Callable[..., T]], Callable[P, T]]:
    """
    Copy the docstring from the provided function, converting it to method-style.
    """

    def wrapped(method: Callable[..., T]) -> Callable[P, T]:
        method_doc = proxy_func.__doc__
        if method.__qualname__.startswith("Graph."):
            method_doc = func_to_method_doc(
                method_doc, GRAPH_METHODS, convert_nx_alias=True
            )
        elif method.__qualname__.startswith("Framework."):
            method_doc = func_to_method_doc(
                method_doc, FRAMEWORK_METHODS, convert_nx_alias=False
            )

        method.__doc__ = method_doc
        method._wrapped_func = proxy_func
        return method

    return wrapped


# Dynamically collect the set of public method names defined on each class.
def _collect_public_methods(class_name: str, rel_file: str) -> set[str]:
    """Return the set of public method names defined in a class."""
    file_path = Path(__file__).resolve().parents[1] / rel_file
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    methods = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for class_node in node.body:
                if not isinstance(class_node, ast.FunctionDef):
                    continue
                if class_node.name.startswith("_"):
                    continue
                # Skip classmethods; they are called as Graph/Framework.method(),
                # not instance.method().
                is_classmethod = any(
                    isinstance(d, ast.Name) and d.id == "classmethod"
                    for d in class_node.decorator_list
                )
                if is_classmethod:
                    continue
                methods.add(class_node.name)
    return methods


GRAPH_METHODS = _collect_public_methods("Graph", "graph/graph.py")
FRAMEWORK_METHODS = _collect_public_methods("Framework", "framework/framework.py")


def _transform_doctest_func_to_method(line: str, class_methods: set[str]) -> str:
    """
    Transform a single doctest line from function-style to method-style.

    ``>>> is_rigid(G)``            -> ``>>> G.is_rigid()``
    ``>>> is_rigid(G, dim=2)``     -> ``>>> G.is_rigid(dim=2)``
    ``>>> is_linked(H, 1,7)``      -> ``>>> H.is_linked(1,7)``
    ``>>> H = zero_extension(G, ...)`` -> ``>>> H = G.zero_extension(...)``
    ``>>> rigid_dim = max_rigid_dimension(G); rigid_dim``
        -> ``>>> rigid_dim = G.max_rigid_dimension(); rigid_dim``
    ``>>> len(list(all_k_extensions(G, 0)))``
        -> ``>>> len(list(G.all_k_extensions(0)))``
    ``>>> type(all_extensions(G))``
        -> ``>>> type(G.all_extensions())``

    Does NOT transform:
    - ``>>> print(G)``         — print is NOT in class_methods
    - ``>>> Graph([(0,1)])``   — Graph is NOT in class_methods
    - ``>>> graphs.Complete(5)`` — Complete is NOT in class_methods
    """
    # Match: >>> [optional_assignment] func_name(var[, args]) [optional_suffix]
    m = re.match(
        r"^(\s*>>> )"  # group 1: prefix with >>>
        r"((?:\w+\s*=\s*)?)"  # group 2: optional assignment
        r"(\w+)"  # group 3: function name
        r"\("  # opening paren
        r"(\w+)"  # group 4: first argument (the graph variable)
        r"(?:,\s*(.*?))?"  # group 5: remaining args (optional, after comma)
        r"\)"  # closing paren
        r"(.*)"  # group 6: suffix
        r"$",
        line,
    )
    if m:
        prefix, assignment, func, var, rest_args, suffix = m.groups()
        if func in class_methods:
            if rest_args:
                return f"{prefix}{assignment}{var}.{func}({rest_args}){suffix}"
            else:
                return f"{prefix}{assignment}{var}.{func}(){suffix}"

    # Fallback: re.sub for wrapped calls
    if not re.search(r"^\s*(?:>>>|\.\.\.)\s", line):
        return line

    def _replace(m):
        func, var, rest = m.group(1), m.group(2), m.group(3)
        if func not in class_methods:
            return m.group(0)
        return f"{var}.{func}({rest})" if rest else f"{var}.{func}()"

    return re.sub(r"(?<!\.)(\w+)\(([A-Za-z_]\w*)(?:,\s*(.*?))?\)", _replace, line)


def _transform_doctest_nx_to_graph_alias(line: str) -> str:
    """
    Transform a single doctest line from NetworkX style to Graph alias style.

    ``>>> G.add_node(3)``            -> ``>>> G.add_vertex(3)``
    ``>>> H.remove_edges_from(E)``   -> ``>>> H.delete_edges(E)``
    ``>>> nx.is_isomorphic(G, H)``   -> ``>>> G.is_isomorphic(H)``
    """
    method_match = re.match(
        r"^(\s*(?:>>>|\.\.\.)\s+)"  # group 1: prefix with >>> or ...
        r"((?:\w+\s*=\s*)?)"  # group 2: optional assignment
        r"(\w+)"  # group 3: variable name (G, H, etc.)
        r"\."  # the dot
        r"(\w+)"  # group 4: method name
        r"\(([^)]*)\)"  # group 5: arguments
        r"(.*)"  # group 6: suffix
        r"$",
        line,
    )
    if method_match:
        prefix, assignment, var, method, args, suffix = method_match.groups()
        graph_alias = GRAPH_ALIAS_NX_TO_METHOD.get(method)
        if graph_alias is not None:
            return f"{prefix}{assignment}{var}.{graph_alias}({args}){suffix}"

    isomorphic_match = re.match(
        r"^(\s*(?:>>>|\.\.\.)\s+)"  # group 1: prefix with >>> or ...
        r"((?:\w+\s*=\s*)?)"  # group 2: optional assignment
        r"(?:nx|networkx)\.is_isomorphic\("  # function name with module prefix
        r"(\w+)"  # group 3: graph variable (self)
        r"(?:,\s*(.*))?"  # group 4: remaining args (greedy to handle nested parens)
        r"\)"
        r"(.*)"  # group 5: suffix
        r"$",
        line,
    )
    if isomorphic_match:
        prefix, assignment, var, rest_args, suffix = isomorphic_match.groups()
        if rest_args:
            return f"{prefix}{assignment}{var}.is_isomorphic({rest_args}){suffix}"

    return line


def _transform_func_to_meth_refs(line: str, class_methods: set[str]) -> str:
    """
    Transform :func: references to :meth: for bare method names.

    ``:func:`.is_rigid```        -> ``:meth:`.is_rigid```
    ``:func:`~.is_rigid```       -> ``:meth:`~.is_rigid```

    Does NOT transform references to non-Graph functions:
    ``:func:`networkx.is_connected```   — not in class_methods
    ``:func:`_build_pebble_digraph```   — not in class_methods
    """

    def _replace(m):
        prefix = m.group(1)
        name = m.group(2)
        if name in class_methods:
            return f":meth:`{prefix}{name}`"
        return m.group(0)

    return re.sub(
        r":func:`([~]?\.?)([A-Za-z_]\w*)(?:\(\))?`",
        _replace,
        line,
    )


def _transform_multiline_doctest_func_to_method(
    docstring: str, class_methods: set[str]
) -> str:
    """
    Transform multiline function-style doctests where the graph variable
    appears alone on the first continuation line.

    ``>>> print(func_name(``    ->   ``>>> print(G.func_name(``
    ``...     G,``                   ``...     arg,``
    ``...     arg,``
    """

    def _replace(m):
        prefix, func, var = m.group(1), m.group(2), m.group(3)
        if func not in class_methods:
            return m.group(0)
        return f"{prefix}{var}.{func}(\n"

    return re.sub(
        r"([ \t]*>>> (?:[^(\n]*\()*)"  # >>> and any outer wrappers, e.g. print(
        r"(\w+)\(\n"  # func_name(  + newline
        r"[ \t]*\.\.\. +(\w+),\n",  # ...     G,  + newline (graph var alone)
        _replace,
        docstring,
        flags=re.MULTILINE,
    )


def func_to_method_doc(
    docstring: str, class_methods: set[str], convert_nx_alias: bool = True
) -> str:
    """
    Convert a function-style docstring to method-style.

    This is what ``copy_doc`` will call.

    Transformations applied:
    1. Multiline doctest: ``>>> func(`` / ``...  G,`` -> ``>>> G.func(``
    2. Doctest: ``>>> method(G, args)`` -> ``>>> G.method(args)``
    3. Doctest aliases: ``>>> G.add_node(v)`` -> ``>>> G.add_vertex(v)``
       (only if ``convert_nx_alias=True``)
    4. ``:func:`.name``` -> ``:meth:`.name``` (bare names only)
    5. ``this function`` -> ``this method``
    6. ``the function`` -> ``the method``
    7. ``The function`` -> ``The method``
    8. Remove bare ``graph:`` / ``framework:`` first entry from Parameters section.
    """
    if docstring is None:
        return None

    docstring = _transform_multiline_doctest_func_to_method(docstring, class_methods)

    lines = docstring.split("\n")
    result = []

    in_params = False
    after_dashes = False
    for line in lines:
        stripped = line.strip()

        # Track when we enter the Parameters section and its dashes line.
        if stripped == "Parameters":
            in_params = True
            after_dashes = False
        elif in_params and not after_dashes and re.match(r"^-+$", stripped):
            after_dashes = True
        elif in_params and after_dashes:
            # Skip a bare 'graph:' or 'framework:' entry (no description follows).
            if re.match(r"^(?:graph|framework):$", stripped):
                in_params = False
                after_dashes = False
                continue
            else:
                in_params = False
                after_dashes = False

        # 1. Doctest lines
        new_line = _transform_doctest_func_to_method(line, class_methods)
        if new_line != line:
            result.append(new_line)
            continue

        # 2. Convert NetworkX alias examples back to Graph alias methods
        #    (only relevant for Graph, not Framework).
        if convert_nx_alias:
            new_line = _transform_doctest_nx_to_graph_alias(line)
            if new_line != line:
                result.append(new_line)
                continue

        # 3. :func: -> :meth: for bare names
        line = _transform_func_to_meth_refs(line, class_methods)

        # 4. "this function" / "the function" / "The function"
        line = re.sub(r"\bthis function\b", "this method", line)
        line = re.sub(r"\bThis function\b", "This method", line)
        line = re.sub(r"\bthe function\b", "the method", line)
        line = re.sub(r"\bThe function\b", "The method", line)

        result.append(line)

    return "\n".join(result)


# ==================== functions for generating package structure ===================


def _contains_py_files(path: str):
    """Check if a directory or its subdirectories contain .py files."""
    for _, _, files in os.walk(path):
        if any(file.endswith(".py") for file in files):
            return True
    return False


def _get_comment_for_file(
    file_path: str, root_dir: str, comment_dict: dict[str, dict[str, str]]
) -> str:
    """
    Resolve a comment for a given file path using nested comment dict.

    Parameters
    ----------
    file_path:
        Full file path.
    root_dir:
        Base directory.
    comment_dict:
        Nested comment dictionary.
    """
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)

    if len(parts) == 1:
        return comment_dict.get(".", {}).get(parts[0], "")
    elif parts[0] in comment_dict:
        return comment_dict[parts[0]].get(parts[-1], "")
    return ""


def generate_myst_tree(
    root_path: str,
    comments: dict[str, dict[str, str]] = None,
    indent: int = 0,
    base_path: str = None,
    show_line_numbers: bool = False,
) -> str:
    """
    Generate MyST-compatible Markdown tree showing only ``.py`` files, folders first.

    Parameters
    ----------
    root_path:
        Current directory to process.
    comments:
        Nested dict ``{folder: {file: comment}}``.
    indent:
        Indentation level.
    base_path:
        Root directory for relative paths.
    show_line_numbers:
        Whether to show line numbers.
    """
    if comments is None:
        comments = {}
    if base_path is None:
        base_path = root_path

    output = []
    entries = sorted(os.listdir(root_path))

    # Separate folders and Python files
    folders = [
        e
        for e in entries
        if os.path.isdir(os.path.join(root_path, e))
        and _contains_py_files(os.path.join(root_path, e))
    ]
    py_files = [
        e
        for e in entries
        if e.endswith(".py") and os.path.isfile(os.path.join(root_path, e))
    ]

    # List folders first
    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        prefix = "    " * indent
        output.append(f"{prefix}{folder}/")
        sub_tree = generate_myst_tree(
            folder_path,
            comments,
            indent + 1,
            base_path,
            show_line_numbers=show_line_numbers,
        )
        output.append(sub_tree)

    # Then list .py files
    for file in py_files:
        file_path = os.path.join(root_path, file)
        if show_line_numbers:
            with open(file_path, "r") as f:
                n = len(f.readlines())

        prefix_and_file = "    " * indent + f"{file} "
        comment = _get_comment_for_file(file_path, base_path, comments)
        num_dots = 80 - len(prefix_and_file) - len(comment)
        comment_str = "." * num_dots + f" {comment}" if comment else ""
        output.append(
            prefix_and_file
            + comment_str
            + (f" ({n:4d} lines)" if show_line_numbers else "")
        )

    return "\n".join(output)


# ======================functions for generating tables of methods ===============


def doc_category(category):
    """
    Decorator for doc categories.
    """

    def decorator_doc_category(func):
        setattr(func, "_doc_category", category)
        return func

    return decorator_doc_category


def generate_category_tables(
    cls, tabs, cat_order=None, include_all=False, add_attributes=True
) -> str:
    """
    Generate a formatted string that categorizes methods of a given class.

    Parameters
    ----------
    cls:
        A class.
    tabs:
        The number of indentation levels that are applied to the output.
    cat_order:
        Optional list specifying the order in which categories appear
        in the output.
    include_all:
        Optional boolean determining whether methods without a specific category
        should be included.
    add_attributes:
        Optional boolean determining whether the public attributes should
        be listed among attribute getters.
    """
    if cat_order is None:
        cat_order = []
    categories = {}
    for func in dir(cls):
        if callable(getattr(cls, func)) and func[:2] != "__":
            f = getattr(cls, func)
            if hasattr(f, "_doc_category"):
                category = f._doc_category
                if category in categories:
                    categories[category].append(func)
                else:
                    categories[category] = [func]
            elif include_all:
                if "Not categorized" in categories:
                    categories["Not categorized"].append(func)
                else:
                    categories["Not categorized"] = [func]
        elif isinstance(getattr(cls, func), property) and add_attributes:
            category = "Attribute getters"
            if category in categories:
                categories[category].append(func)
            else:
                categories[category] = [func]

    for category in categories:
        if category not in cat_order:
            cat_order.append(category)

    res = "Methods\n-------\n"
    for category, functions in sorted(
        categories.items(), key=lambda t: cat_order.index(t[0])
    ):
        res += f"**{category}**"
        res += "\n\n.. autosummary::\n\n    "
        res += "\n    ".join(functions)
        res += "\n\n"
    indent = "".join(["    " for _ in range(tabs)])
    return ("\n" + indent).join(res.splitlines())
