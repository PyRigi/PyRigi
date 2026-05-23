from pyrigi._utils._doc import (
    GRAPH_METHODS,
    _transform_doctest_func_to_method,
    _transform_func_to_meth_refs,
    func_to_method_doc,
)


def test_func_to_method_doc_rewrites_networkx_alias_methods():
    doc = "\n".join(
        [
            ">>> G.add_node(3)",
            ">>> H.remove_nodes_from([1, 2])",
            ">>> H.remove_edge(1, 2)",
            ">>> H.remove_edges_from(E)",
            ">>> out = H.add_edges_from(E)",
        ]
    )

    transformed = func_to_method_doc(doc, class_methods=set())

    assert ">>> G.add_vertex(3)" in transformed
    assert ">>> H.delete_vertices([1, 2])" in transformed
    assert ">>> H.delete_edge(1, 2)" in transformed
    assert ">>> H.delete_edges(E)" in transformed
    assert ">>> out = H.add_edges(E)" in transformed


def test_func_to_method_doc_rewrites_networkx_isomorphic():
    doc = "\n".join(
        [
            ">>> nx.is_isomorphic(G, H)",
            ">>> answer = networkx.is_isomorphic(G, H)",
            ">>> nx.is_connected(G)",
        ]
    )

    transformed = func_to_method_doc(doc, class_methods=set())

    assert ">>> G.is_isomorphic(H)" in transformed
    assert ">>> answer = G.is_isomorphic(H)" in transformed
    assert ">>> nx.is_connected(G)" in transformed


def test_func_to_method_doc_keeps_unmapped_graph_methods():
    doc = ">>> G.add_edge(0, 1)"

    transformed = func_to_method_doc(doc, class_methods=set())

    assert transformed == doc


def test_transform_doctest_func_to_method_no_args():
    line = ">>> is_rigid(G)"
    result = _transform_doctest_func_to_method(line, {"is_rigid"})
    assert result == ">>> G.is_rigid()"


def test_transform_doctest_func_to_method_with_args():
    line = ">>> is_rigid(G, dim=2)"
    result = _transform_doctest_func_to_method(line, {"is_rigid"})
    assert result == ">>> G.is_rigid(dim=2)"


def test_transform_doctest_func_to_method_with_assignment():
    line = ">>> H = zero_extension(G, 3, 4)"
    result = _transform_doctest_func_to_method(line, {"zero_extension"})
    assert result == ">>> H = G.zero_extension(3, 4)"


def test_transform_doctest_func_to_method_skips_non_class_method():
    line = ">>> print(G)"
    result = _transform_doctest_func_to_method(line, {"is_rigid"})
    assert result == line


def test_transform_func_to_meth_refs_bare_dot():
    line = ":func:`.is_rigid`"
    result = _transform_func_to_meth_refs(line, {"is_rigid"})
    assert result == ":meth:`.is_rigid`"


def test_transform_func_to_meth_refs_tilde_dot():
    line = ":func:`~.is_rigid`"
    result = _transform_func_to_meth_refs(line, {"is_rigid"})
    assert result == ":meth:`~.is_rigid`"


def test_transform_func_to_meth_refs_keeps_non_class_method():
    line = ":func:`networkx.is_connected`"
    result = _transform_func_to_meth_refs(line, {"is_rigid"})
    assert result == line


def test_func_to_method_doc_rewrites_func_refs():
    doc = "See :func:`~.is_rigid` and :func:`~.is_linked`."
    result = func_to_method_doc(doc, {"is_rigid", "is_linked"})
    assert ":meth:`~.is_rigid`" in result
    assert ":meth:`~.is_linked`" in result


def test_func_to_method_doc_rewrites_method_text():
    doc = "this function does X. The function returns Y. Use the function carefully."
    result = func_to_method_doc(doc, set())
    assert "this method" in result
    assert "The method" in result
    assert "the method" in result


def test_func_to_method_doc_on_real_docstring():
    doc = """
    Return whether a pair of vertices is ``dim``-linked.

    Examples
    --------
    >>> H = Graph([[0, 1], [0, 2], [1, 3], [1, 5], [2, 3], [2, 6], [3, 5], [3, 7], [5, 7], [6, 7], [3, 6]])  # noqa: E501
    >>> is_linked(H, 1,7)
    True
    >>> H = Graph([[0, 1], [0, 2], [1, 3], [2, 3]])
    >>> is_linked(H, 0,3)
    False
    >>> is_linked(H, 1,3)
    True
    """

    transformed = func_to_method_doc(doc, GRAPH_METHODS)

    assert ">>> H.is_linked(1,7)" in transformed
    assert ">>> H.is_linked(0,3)" in transformed
    assert ">>> H.is_linked(1,3)" in transformed
    assert ">>> is_linked(" not in transformed


def test_func_to_method_doc_rewrites_doctest_lines():
    doc = "\n".join(
        [
            ">>> is_rigid(G)",
            ">>> is_rigid(G, dim=2)",
            ">>> H = zero_extension(G, 3, 4)",
        ]
    )
    result = func_to_method_doc(doc, {"is_rigid", "zero_extension"})
    assert ">>> G.is_rigid()" in result
    assert ">>> G.is_rigid(dim=2)" in result
    assert ">>> H = G.zero_extension(3, 4)" in result
