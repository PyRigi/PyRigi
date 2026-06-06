import networkx as nx

from pyrigi.graphDB import GraphStoreService, QueryFilter


def main() -> None:
    # 1. Setup
    store = GraphStoreService("pyrigi/graphDB/outputs/graph_store.db").init()
    store.ingest("pyrigi/graphDB/outputs/g6")
    print(store.count(), "graphs ingested")

    # 2. Populate built-in rigidity columns
    store.populate_column("min_rigidity")
    store.populate_column("rigidity")
    store.populate_column("global_rigidity")

    # 3. Basic fetch — filters, ordering, pretty-print
    store.pretty_print_results(
        store.fetch(
            select=[
                "graph",
                "num_vertices",
                "num_edges",
                "rigidity",
                "min_rigidity",
                "global_rigidity",
            ],
            filters=[QueryFilter("num_vertices", "=", 5)],
            order_by="num_edges",
        ),
        show_index=True,
        max_rows=5,
    )

    # 4. Rigidity-aware queries
    # = 2: fetch strategy expands to (rigidity = 2 OR rigidity = -1), so complete
    # graphs (stored as -1) are included alongside graphs with max_rigid_dim = 2
    store.pretty_print_results(
        store.fetch(
            select=["graph", "num_vertices", "rigidity"],
            filters=[QueryFilter("rigidity", "=", 2)],
            limit=5,
        ),
        show_index=True,
    )

    # IN [1, 2]: expands to (rigidity IN (1, 2) OR rigidity = -1),
    # including complete graphs
    store.pretty_print_results(
        store.fetch(
            select=["graph", "num_vertices", "rigidity"],
            filters=[QueryFilter("rigidity", "IN", [1, 2])],
            limit=5,
        ),
        show_index=True,
    )

    # = 2 on min_rigidity: includes non-complete minimally 2-rigid graphs and K3
    # (complete, stored as -2, satisfies -2 >= -2)
    store.pretty_print_results(
        store.fetch(
            select=["graph", "num_vertices", "min_rigidity"],
            filters=[QueryFilter("min_rigidity", "=", 2)],
            limit=5,
        ),
        show_index=True,
    )

    # 5. Custom column + fluent query builder
    store.add_column(
        "density",
        "REAL",
        populator=lambda row: (
            row["num_edges"] / (row["num_vertices"] * (row["num_vertices"] - 1) / 2)
        ),
    )
    store.populate_column("density")

    (
        store.query()
        .select(["graph", "num_vertices", "density", "rigidity"])
        .where_any(
            [QueryFilter("num_vertices", "=", 5), QueryFilter("num_vertices", "=", 7)]
        )
        .order_by("density", asc=False)
        .limit(5)
        .pretty_print(show_index=True)
    )

    # 6. Streaming with a mapper — yields networkx Graph objects
    for g in store.iter_fetch(
        select=["graph"],
        filters=[QueryFilter("num_vertices", "=", 5)],
        limit=3,
        mapper=lambda row: nx.from_graph6_bytes(row["graph"].encode("ascii")),
    ):
        print(g.number_of_nodes(), "vertices,", g.number_of_edges(), "edges")

    # 7. Correcting a column mistake with drop_column
    # Suppose a column is added with wrong logic.
    store.add_column(
        "is_tree",
        "INTEGER",
        description="1 if the graph is a tree, 0 otherwise",
        populator=lambda row: int(row["num_edges"] == row["num_vertices"]),  # wrong
    )
    store.populate_column("is_tree")

    # Realise the mistake: a tree satisfies num_edges == num_vertices - 1.
    # Drop the column and re-add it with the correct populator.
    store.drop_column("is_tree")
    store.add_column(
        "is_tree",
        "INTEGER",
        description="1 if the graph is a tree, 0 otherwise",
        populator=lambda row: int(
            row["num_edges"] == row["num_vertices"] - 1
        ),  # correct
    )
    store.populate_column("is_tree")
    print(
        "is_tree column corrected:",
        store.fetch(select=["num_vertices", "is_tree"], limit=3),
    )

    # 8. Re-registering a fetch strategy with update_column_fetch_strategy
    # Runtime callables (lambdas) are not persisted across sessions.
    store.update_column_fetch_strategy(
        "density",
        fetch_strategy=lambda col, op, val: (f"{col} {op} ?", [val]),
    )
    print("Fetch strategy re-registered for density column")

    store.close()


if __name__ == "__main__":
    main()
