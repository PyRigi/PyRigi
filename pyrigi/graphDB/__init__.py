"""
pyrigi.graphDB
~~~~~~~~~~~~~~
Graph generators and extensible SQLite graph database interface.

Graph generators (replacing the former ``pyrigi/graphDB.py`` module)::

    import pyrigi.graphDB as graphs
    G = graphs.Complete(5)

SQLite graph database quick start::

    from pyrigi.graphDB import GraphStoreService, QueryFilter

    store = GraphStoreService("outputs/graph_store.db").init()
    store.ingest("outputs/g6")

    rows = store.fetch(
        select=["graph", "num_vertices", "min_rigidity"],
        filters=[
            QueryFilter("num_vertices", "=", 7),
            QueryFilter("min_rigidity", "=", 3),
        ],
    )
"""

# Graph generators (formerly pyrigi/graphDB.py)
from pyrigi.graphDB.small_graphs import (
    Cycle,
    Complete,
    CompleteLooped,
    Path,
    CompleteBipartite,
    K33plusEdge,
    Diamond,
    DiamondWithZeroExtension,
    ThreePrism,
    ThreePrismPlusEdge,
    ThreePrismPlusTriangleOnSide,
    CubeWithDiagonal,
    DoubleBanana,
    CompleteMinusOne,
    Octahedral,
    Icosahedral,
    Dodecahedral,
    Frustum,
    K66MinusPerfectMatching,
    CnSymmetricFourRegular,
    CnSymmetricWithFixedVertex,
    ThreeConnectedR3Circuit,
    Wheel,
    Grid,
)

# Re-exported for test access via `pyrigi.graphDB._min_rigidity_dimension_encoding`
from pyrigi.graphDB.small_graphs import _min_rigidity_dimension_encoding  # noqa: F401

# SQLite graph database interface
from pyrigi.graphDB.models import (
    AndExpr,
    ColumnDef,
    IngestStats,
    NotExpr,
    OrExpr,
    PopulateStats,
    QueryExpr,
    QueryFilter,
    all_of,
    any_of,
    not_,
)
from pyrigi.graphDB.query import QueryBuilder
from pyrigi.graphDB.service import GraphStoreService

__all__ = [
    # Graph generators
    "Cycle",
    "Complete",
    "CompleteLooped",
    "Path",
    "CompleteBipartite",
    "K33plusEdge",
    "Diamond",
    "DiamondWithZeroExtension",
    "ThreePrism",
    "ThreePrismPlusEdge",
    "ThreePrismPlusTriangleOnSide",
    "CubeWithDiagonal",
    "DoubleBanana",
    "CompleteMinusOne",
    "Octahedral",
    "Icosahedral",
    "Dodecahedral",
    "Frustum",
    "K66MinusPerfectMatching",
    "CnSymmetricFourRegular",
    "CnSymmetricWithFixedVertex",
    "ThreeConnectedR3Circuit",
    "Wheel",
    "Grid",
    # SQLite database interface
    "GraphStoreService",
    "QueryFilter",
    "AndExpr",
    "OrExpr",
    "NotExpr",
    "QueryExpr",
    "all_of",
    "any_of",
    "not_",
    "ColumnDef",
    "IngestStats",
    "PopulateStats",
    "QueryBuilder",
]
