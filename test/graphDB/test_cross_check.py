"""Tests cross-checking databases of rigid graphs."""

import networkx as nx
import pytest

from pyrigi.graph._rigidity.generic import is_rigid
from pyrigi.graphDB import GraphStoreService, QueryFilter
from pyrigi.graphDB.models import not_

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


@pytest.fixture(params=[6], scope="session")
def store_connected(request):
    """In-memory store of connected graphs."""
    store = GraphStoreService(":memory:").init()
    for n in range(2, request.param + 1):
        store.ingest(f"test/graphDB/connected_graphs/graphs_{n}.g6")
    store.populate_column("rigidity")
    yield store
    store.close()


def test_populate_rigidity(store_connected):
    rows = store_connected.fetch(
        filters=[QueryFilter("rigidity", "IS NULL")],
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


def test_rigid_nums_computed(store_connected):
    num_vert = max(
        [row["num_vertices"] for row in store_connected.fetch(select=["num_vertices"])]
    )
    for dim in range(1, num_vert + 1):
        for n in range(2, num_vert + 1):
            rows = store_connected.iter_fetch(
                filters=[QueryFilter("num_vertices", "=", n)], select=["graph"]
            )
            cnt = 0
            for row in rows:
                G = nx.from_graph6_bytes(row["graph"].encode())
                if is_rigid(G, dim=dim):
                    cnt += 1
            assert cnt == DIM_RIGID_NUMS[dim][n]


def test_fetched_rigid_are_rigid(store_connected):
    for dim in range(1, 6):
        rows = store_connected.iter_fetch(
            filters=[QueryFilter("rigidity", "=", dim)], select=["graph"]
        )
        for row in rows:
            G = nx.from_graph6_bytes(row["graph"].encode())
            assert is_rigid(
                G, dim=dim
            ), f"Graph {G} was fetched as {dim}-rigid, but it is {dim}-rigid."


def test_fetched_flexible_are_flexible(store_connected):
    for dim in range(1, 6):
        rows = store_connected.iter_fetch(
            expr=not_(QueryFilter("rigidity", "=", dim)), select=["graph"]
        )
        for row in rows:
            G = nx.from_graph6_bytes(row["graph"].encode())
            assert not is_rigid(
                G, dim=dim
            ), f"Graph {G} was fetched as {dim}-flexible, but it is {dim}-rigid."
