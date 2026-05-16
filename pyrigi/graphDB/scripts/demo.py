import networkx as nx

from pyrigi.graphDB import GraphStoreService, QueryFilter

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
# >= 2 includes complete graphs (stored as NULL — rigid in all dimensions)
store.pretty_print_results(
    store.fetch(
        select=["graph", "num_vertices", "rigidity"],
        filters=[QueryFilter("rigidity", ">=", 2)],
        limit=5,
    ),
    show_index=True,
)

# = 2 includes complete graphs that are minimally 2-rigid (e.g. K3, stored as -2)
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

store.close()
