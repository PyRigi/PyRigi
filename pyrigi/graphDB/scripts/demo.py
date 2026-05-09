import networkx as nx

from pyrigi.graphDB import AndExpr, GraphStoreService, OrExpr, QueryFilter

# -- Init (creates pyrigi/graphDB/outputs/graph_store.db)
store = GraphStoreService("pyrigi/graphDB/outputs/graph_store.db").init()

# -- Ingest all g6 files
stats = store.ingest("pyrigi/graphDB/outputs/g6")
print(stats)

# -- Populate rigidity columns
store.populate_column("min_rigidity")
store.populate_column("rigidity")
store.populate_column("global_rigidity")

# -- Basic sanity
print(store.count())       # total graphs
print(store.info())        # all columns + whether they have a populator

# -- Query: all 5-vertex graphs
rows = store.fetch(
    select=["graph", "num_vertices", "num_edges", "min_degree", "max_degree"],
    filters=[QueryFilter("num_vertices", "=", 5)],
    order_by="num_edges",
)
print(f"{len(rows)} five-vertex graphs")
print(rows[0])

# -- Add a custom column
store.add_column("density", "REAL",
    populator=lambda row: row["num_edges"] / (row["num_vertices"] * (row["num_vertices"] - 1) / 2))
pstats = store.populate_column("density")
print(pstats)

# -- Query custom column
rows = store.fetch(
    select=["graph", "num_vertices", "density"],
    filters=[QueryFilter("num_vertices", "=", 7)],
    order_by="density",
    limit=10,
)
for r in rows:
    print(r)

# -- Grouped boolean logic with expression tree
rows = store.fetch(
    select=["graph", "num_vertices", "density"],
    expr=AndExpr([
        OrExpr([
            QueryFilter("num_vertices", "=", 5),
            QueryFilter("num_vertices", "=", 7),
        ]),
        OrExpr([
            QueryFilter("density", ">=", 0.5),
            QueryFilter("density", "IS NULL", None),
        ]),
    ]),
    order_by="density",
    limit=10,
)

store.pretty_print_results(
    rows,
    show_index=True,
    max_rows=10,
)
print(f"Grouped fetch returned {len(rows)} rows")

# -- Fluent builder grouped equivalent using OR helper groups
rows = (
    store.query()
    .select(["graph", "num_vertices", "density"])
    .where_any([
        QueryFilter("num_vertices", "=", 5),
        QueryFilter("num_vertices", "=", 7),
    ])
    .where_any([
        QueryFilter("density", ">=", 0.5),
        QueryFilter("density", "IS NULL", None),
    ])
    .order_by("density", asc=False)
    .limit(5)
    .fetch()
)
print(rows)

# -- Pretty print already-fetched rows
store.pretty_print_results(rows, max_rows=5, show_index=True)

# -- Pretty print directly from fluent query
(
    store.query()
    .select(["num_vertices", "num_edges", "density"])
    .order_by("density", asc=False)
    .limit(5)
    .pretty_print(show_index=True)
)

# -- Streaming results as networkx.Graph objects
def to_nx_graph(row: dict) -> nx.Graph:
    return nx.from_graph6_bytes(row["graph"].encode("ascii"))


graphs_iter = store.iter_fetch(
    select=["graph"],
    filters=[QueryFilter("num_vertices", "=", 7)],
    limit=3,
    mapper=to_nx_graph,
)
for g in graphs_iter:
    print("streamed graph:", g.number_of_nodes(), g.number_of_edges())

# -- What rigidity looks like before you plug in a solver
print(store.fetch(select=["graph", "rigidity"], limit=3))

# -- Simulate plugging in a solver (placeholder returns a fixed value)
store.update_column_populator("rigidity", populator=lambda row: 2)
store.populate_column("rigidity")
print(store.fetch(select=["rigidity"], limit=3))
