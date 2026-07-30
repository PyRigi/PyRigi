"""Shared mappings between PyRigi Graph aliases and NetworkX methods."""

GRAPH_ALIAS_METHOD_TO_NX: dict[str, str] = {
    "add_vertex": "add_node",
    "add_vertices": "add_nodes_from",
    "add_edges": "add_edges_from",
    "delete_vertex": "remove_node",
    "delete_vertices": "remove_nodes_from",
    "delete_edge": "remove_edge",
    "delete_edges": "remove_edges_from",
}

GRAPH_ALIAS_NX_TO_METHOD: dict[str, str] = {
    nx_method: graph_method
    for graph_method, nx_method in GRAPH_ALIAS_METHOD_TO_NX.items()
}
