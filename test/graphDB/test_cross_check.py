"""Tests cross-checking databases of rigid graphs."""

import networkx as nx
from pathlib import Path
import pytest

from pyrigi.graph._rigidity.generic import is_min_rigid, is_rigid
from pyrigi.graph._rigidity.global_ import is_globally_rigid
from pyrigi.graphDB import GraphStoreService, QueryFilter
from pyrigi.graphDB.models import not_


RIGIDITY_TYPES = ["rigidity", "min_rigidity", "global_rigidity"]
RIGIDITY_FUNCTIONS = [is_rigid, is_min_rigid, is_globally_rigid]

CONNECTED_NUMS = [0, 1, 1, 2, 6, 21, 112, 853, 11117, 261080, 11716571]
# CONNECTED_NUMS[i] is the number of connected graphs with i vertices
# https://oeis.org/A001349


DIM_RIGID_NUMS = {
    d + 1: row
    for d, row in enumerate(
        [
            [0, 1, 1, 2, 6, 21, 112, 853, 11117],
            [0, 1, 1, 1, 2, 7, 42, 377, 6199, 180878],
            [0, 1, 1, 1, 1, 2, 8, 61, 1054, 41304, 3242050],
            [0, 1, 1, 1, 1, 1, 2, 8, 76, 2148, 192127],
            [0, 1, 1, 1, 1, 1, 1, 2, 8, 83, 3328, 581909],
            [0, 1, 1, 1, 1, 1, 1, 1, 2, 8, 87, 4223, 1222742],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 88, 4751, 1941123],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 89, 5003, 2529860],
        ]
    )
}
# DIM_RIGID_NUMS[dim][n] is the number of dim-graphs with n vertices


@pytest.fixture(
    params=[
        6,
        pytest.param(7, marks=pytest.mark.long_local),
    ],
    scope="session",
)
def store_connected(request):
    """In-memory store of connected graphs."""
    store = GraphStoreService(":memory:").init()
    for n in range(2, request.param + 1):
        store.ingest(str(Path(__file__).parent / "connected_graphs" / f"graphs_{n}.g6"))

    for rigidity_type in RIGIDITY_TYPES:
        store.populate_column(rigidity_type)

    yield store
    store.close()


@pytest.mark.parametrize(
    "rigidity_type",
    RIGIDITY_TYPES,
)
def test_populating(store_connected, rigidity_type):
    rows = store_connected.fetch(
        filters=[QueryFilter(rigidity_type, "IS NULL")],
    )
    assert len(rows) == 0


def test_connected_nums(store_connected):
    num_vert = max(
        [row["num_vertices"] for row in store_connected.fetch(select=["num_vertices"])]
    )
    for n in range(2, num_vert + 1):
        rows = store_connected.fetch(
            filters=[QueryFilter("num_vertices", "=", n)],
        )
        assert len(rows) == CONNECTED_NUMS[n], f"n={n}"


def test_rigid_nums_fetched(store_connected):
    num_vert = max(
        [row["num_vertices"] for row in store_connected.fetch(select=["num_vertices"])]
    )
    for dim in range(1, num_vert + 1):
        for n in range(2, num_vert + 1):
            rows = store_connected.fetch(
                filters=[
                    QueryFilter("num_vertices", "=", n),
                    QueryFilter("rigidity", "=", dim),
                ],
            )
            assert len(rows) == DIM_RIGID_NUMS[dim][n]


# def test_rigid_nums_computed(store_connected):
#     """This is only to verify DIM_RIGID_NUMS."""
#     num_vert = max(
#         [row["num_vertices"] for row in store_connected.fetch(select=["num_vertices"])]
#     )
#     for dim in range(1, num_vert + 1):
#         for n in range(2, num_vert + 1):
#             rows = store_connected.iter_fetch(
#                 filters=[QueryFilter("num_vertices", "=", n)], select=["graph"]
#             )
#             cnt = 0
#             for row in rows:
#                 G = nx.from_graph6_bytes(row["graph"].encode())
#                 if is_rigid(G, dim=dim):
#                     cnt += 1
#             assert cnt == DIM_RIGID_NUMS[dim][n]


@pytest.mark.parametrize(
    "rigidity_type, rigidity_function",
    zip(RIGIDITY_TYPES, RIGIDITY_FUNCTIONS),
)
def test_fetched_rigid_are_rigid(store_connected, rigidity_type, rigidity_function):
    for dim in range(1, 6):
        rows = store_connected.iter_fetch(
            filters=[QueryFilter(rigidity_type, "=", dim)],
            select=["graph", rigidity_type],
        )
        for row in rows:
            G = nx.from_graph6_bytes(row["graph"].encode())
            assert rigidity_function(G, dim=dim), (
                f"Graph with g6 encoding '{row["graph"]}' was fetched "
                f"as {dim}-{rigidity_type.replace("dity", "d")}, but it is not. "
                f"The stored '{rigidity_type}' value is {row[rigidity_type]}."
            )


@pytest.mark.parametrize(
    "rigidity_type, rigidity_function",
    zip(RIGIDITY_TYPES, RIGIDITY_FUNCTIONS),
)
def test_fetched_non_rigid_are_not_rigid(
    store_connected, rigidity_type, rigidity_function
):
    for dim in range(1, 6):
        rows = store_connected.iter_fetch(
            expr=not_(QueryFilter(rigidity_type, "=", dim)),
            select=["graph", rigidity_type],
        )
        for row in rows:
            G = nx.from_graph6_bytes(row["graph"].encode())
            assert not rigidity_function(G, dim=dim), (
                f"Graph with g6 encoding '{row["graph"]}' was fetched as "
                f"not {dim}-{rigidity_type.replace("dity", "d")}, but it is. "
                f"The stored '{rigidity_type}' value is {row[rigidity_type]}."
            )
