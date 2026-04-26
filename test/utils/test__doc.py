from pyrigi._utils._doc import func_to_method_doc


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
