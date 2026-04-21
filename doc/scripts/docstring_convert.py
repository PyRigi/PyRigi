"""
Standalone script for testing docstring conversion between
function-style and method-style.
"""

import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# Dynamically collect the set of public method names defined on the Graph class,
def _collect_graph_methods() -> set[str]:
    """Return the set of public method names defined on the Graph class."""
    from pyrigi import Graph
    import networkx as nx

    nx_methods = set(dir(nx.Graph))
    return {
        name
        for name, _ in inspect.getmembers(Graph, predicate=inspect.isfunction)
        if not name.startswith("_") and name not in nx_methods
    }


GRAPH_METHODS = _collect_graph_methods()


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
    # Pattern: (prefix)(var).(method)( ... ) (possible suffix like "; rigid_dim")
    # We need to handle two cases:
    #   1. Direct call: >>> var.method(args)
    #   2. Assignment:  >>> result = var.method(args)
    #   3. Assignment with suffix: >>> result = var.method(args); suffix

    # Match: >>> [optional_assignment] var.method(args) [optional_suffix]
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


def _transform_doctest_func_to_method(line: str, class_methods: set[str]) -> str:
    """
    Transform a single doctest line from function-style to method-style.

    ``>>> is_rigid(G)``            -> ``>>> G.is_rigid()``
    ``>>> is_rigid(G, dim=2)``     -> ``>>> G.is_rigid(dim=2)``
    ``>>> is_linked(H, 1,7)``      -> ``>>> H.is_linked(1,7)``
    ``>>> H = zero_extension(G, ...)`` -> ``>>> H = G.zero_extension(...)``
    ``>>> rigid_dim = max_rigid_dimension(G); rigid_dim``
        -> ``>>> rigid_dim = G.max_rigid_dimension(); rigid_dim``

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
        prefix = m.group(1)  # e.g. "." or "~." or "~"
        name = m.group(2)  # e.g. "is_rigid"

        if name in class_methods:
            return f":func:`{prefix}{name}`"
        return m.group(0)

    return re.sub(
        r":meth:`([~]?\.?)([A-Za-z_]\w*)(?:\(\))?`",
        _replace,
        line,
    )


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


def func_to_method_doc(docstring: str, class_methods: set[str]) -> str:
    """
    Convert a function-style docstring to method-style.

    This is what ``copy_doc`` will call.

    Transformations applied:
    1. Doctest: ``>>> method(G, args)`` -> ``>>> G.method(args)``
    2. ``:func:`.name``` -> ``:meth:`.name``` (bare names only)
    3. ``this function`` -> ``this method``
    4. ``the function`` -> ``the method``
    5. ``The function`` -> ``The function``
    """
    if docstring is None:
        return None

    lines = docstring.split("\n")
    result = []

    for line in lines:
        # 1. Doctest lines
        new_line = _transform_doctest_func_to_method(line, class_methods)
        if new_line != line:
            result.append(new_line)
            continue

        # 2. :func: -> :meth: for bare names
        line = _transform_func_to_meth_refs(line, class_methods)

        # 3. "this function" / "the function" / "The function"
        line = re.sub(r"\bthis function\b", "this method", line)
        line = re.sub(r"\bthe function\b", "the method", line)
        line = re.sub(r"\bThe function\b", "The method", line)

        result.append(line)

    return "\n".join(result)


if __name__ == "__main__":
    # Test: method->func on rigid_components docstring
    original = """
    Return the list of the vertex sets of ``dim``-rigid components.

    Definitions
    -----
    :prf:ref:`Rigid components <def-rigid-components>`

    Parameters
    ---------
    dim:
        The dimension that is used for the rigidity check.
    algorithm:
        If ``"graphic"`` (only if ``dim=1``),
        then the connected components are returned.

        If ``"subgraphs-pebble"`` (only if ``dim=2``),
        then all subgraphs are checked
        using :meth:`.is_rigid` with ``algorithm="pebble"``.

        If ``"pebble"`` (only if ``dim=2``),
        then :meth:`.Rd_closure` with ``algorithm="pebble"``
        is used.

        If ``"randomized"``, all subgraphs are checked
        using :meth:`.is_rigid` with ``algorithm="randomized"``.

        If ``"numerical"``, all subgraphs are checked
        using :meth:`.is_rigid` with ``algorithm="numerical"``.

        If ``"default"``, then ``"graphic"`` is used for ``dim=1``,
        ``"pebble"`` for ``dim=2``, and ``"randomized"`` for ``dim>=3``.
    prob:
        A bound on the probability for false negatives of the rigidity testing
        when ``algorithm="randomized"``.

        *Warning:* this is not the probability of wrong results in this method,
        but is just passed on to rigidity testing.

    Examples
    --------
    >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
    >>> G.rigid_components(algorithm="randomized")
    [[0, 1], [0, 3], [1, 2], [2, 3]]

    >>> G = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,2), (5,3)])
    >>> G.is_rigid()
    False
    >>> G.rigid_components(algorithm="randomized")
    [[0, 5], [2, 3], [0, 1, 2], [3, 4, 5]]

    Notes
    -----
    If the graph itself is rigid, it is clearly maximal and is returned.
    Every edge is part of a rigid component. Isolated vertices form
    additional rigid components.

    For the pebble game algorithm we use the fact that the ``R2_closure``
    consists of edge disjoint cliques, so we only have to determine them.
    """

    func_style = method_to_func_doc(original, GRAPH_METHODS)
    print("=== FUNCTION STYLE ===")
    print(func_style)

    method_style = func_to_method_doc(func_style, GRAPH_METHODS)
    print("\n=== BACK TO METHOD STYLE ===")
    print(method_style)

    assert method_style == original, "Round-trip failed!"
    print("\nRound-trip passed!")
