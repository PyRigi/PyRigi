"""Tests for custom fetch strategies on the rigidity columns."""

import pytest
import networkx as nx

from pyrigi.graphDB.defaults.fetch_strategies import (
    _min_rigidity_fetch_strategy,
    _rigidity_fetch_strategy,
)
from pyrigi.graphDB import GraphStoreService, QueryFilter


# ---------------------------------------------------------------------------
# Unit tests: _rigidity_fetch_strategy
# ---------------------------------------------------------------------------


class TestRigidityFetchStrategy:
    def test_gte_includes_complete(self):
        sql, params = _rigidity_fetch_strategy("rigidity", ">=", 2)
        assert sql == "(rigidity >= ? OR rigidity = -1)"
        assert params == [2]

    def test_gt_includes_complete(self):
        sql, params = _rigidity_fetch_strategy("rigidity", ">", 1)
        assert sql == "(rigidity > ? OR rigidity = -1)"
        assert params == [1]

    def test_eq_passthrough(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "=", 3)
        assert sql == "rigidity = ?"
        assert params == [3]

    def test_lt_passthrough(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "<", 3)
        assert sql == "rigidity < ?"
        assert params == [3]

    def test_lte_passthrough(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "<=", 2)
        assert sql == "rigidity <= ?"
        assert params == [2]

    def test_is_null_passthrough(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "IS NULL", None)
        assert sql == "rigidity IS NULL"
        assert params == []

    def test_is_not_null_passthrough(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "IS NOT NULL", None)
        assert sql == "rigidity IS NOT NULL"
        assert params == []

    def test_operator_case_insensitive(self):
        sql, params = _rigidity_fetch_strategy("global_rigidity", ">=", 1)
        assert "= -1" in sql
        assert params == [1]


# ---------------------------------------------------------------------------
# Unit tests: _min_rigidity_fetch_strategy
# ---------------------------------------------------------------------------


class TestMinRigidityFetchStrategy:
    def test_eq_expands_for_complete_graphs(self):
        sql, params = _min_rigidity_fetch_strategy("min_rigidity", "=", 2)
        assert sql == "(min_rigidity = ? OR (min_rigidity < 0 AND min_rigidity >= ?))"
        assert params == [2, -2]

    def test_eq_negates_value_correctly(self):
        sql, params = _min_rigidity_fetch_strategy("min_rigidity", "=", 1)
        assert params == [1, -1]

    def test_gt_passthrough(self):
        sql, params = _min_rigidity_fetch_strategy("min_rigidity", ">", 1)
        assert sql == "min_rigidity > ?"
        assert params == [1]

    def test_lt_passthrough(self):
        sql, params = _min_rigidity_fetch_strategy("min_rigidity", "<", 0)
        assert sql == "min_rigidity < ?"
        assert params == [0]

    def test_gte_passthrough(self):
        sql, params = _min_rigidity_fetch_strategy("min_rigidity", ">=", 0)
        assert sql == "min_rigidity >= ?"
        assert params == [0]

    def test_operator_case_insensitive(self):
        sql, params = _min_rigidity_fetch_strategy("min_rigidity", "=", 3)
        assert "min_rigidity = ?" in sql
        assert params == [3, -3]


# ---------------------------------------------------------------------------
# Integration tests via GraphStoreService
# ---------------------------------------------------------------------------


def _g6(g: nx.Graph) -> str:
    return nx.to_graph6_bytes(g, header=False).strip().decode("ascii")


@pytest.fixture
def store_with_rigidity_data(tmp_path):
    """Store ingested with K3 (complete), path P4, and diamond."""
    g6_file = tmp_path / "graphs.g6"
    graphs = [
        nx.complete_graph(3),  # K3: rigidity=-1, min_rigidity=-2
        nx.path_graph(4),  # P4: minimally 1-rigid, min_rigidity=1
        nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),  # diamond: min_rigidity=2
    ]
    lines = [_g6(g) for g in graphs]
    g6_file.write_text("\n".join(lines) + "\n")

    store = GraphStoreService(":memory:").init()
    store.ingest(str(g6_file))
    yield store
    store.close()


class TestRigidityFetchIntegration:
    def test_gte_returns_complete_graph(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("rigidity")
        # K3 has rigidity=-1 (complete); ">= 1" must include it
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("rigidity", ">=", 1)],
        )
        g6_k3 = _g6(nx.complete_graph(3))
        graphs = {r["graph"] for r in rows}
        assert g6_k3 in graphs

    def test_gt_returns_complete_graph(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("rigidity")
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("rigidity", ">", 0)],
        )
        g6_k3 = _g6(nx.complete_graph(3))
        assert any(r["graph"] == g6_k3 for r in rows)

    def test_eq_does_not_return_complete_graph(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("rigidity")
        # "=" queries the raw stored value; complete graphs store -1, so "= 1" must not match them
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("rigidity", "=", 1)],
        )
        g6_k3 = _g6(nx.complete_graph(3))
        assert all(r["graph"] != g6_k3 for r in rows)


class TestMinRigidityFetchIntegration:
    def test_eq_1_returns_path_not_complete(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("min_rigidity")
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("min_rigidity", "=", 1)],
        )
        g6_p4 = _g6(nx.path_graph(4))
        g6_k3 = _g6(nx.complete_graph(3))
        graphs = {r["graph"] for r in rows}
        assert g6_p4 in graphs
        # K3 is minimally 2-rigid (not 1-rigid), so should not appear
        assert g6_k3 not in graphs

    def test_eq_2_returns_diamond_and_complete(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("min_rigidity")
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("min_rigidity", "=", 2)],
        )
        g6_diamond = _g6(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]))
        g6_k3 = _g6(nx.complete_graph(3))
        graphs = {r["graph"] for r in rows}
        assert g6_diamond in graphs
        # K3 is minimally 2-rigid (complete on 3 vertices, n-1=2 <= d=2)
        assert g6_k3 in graphs

    def test_eq_3_returns_complete_not_path_or_diamond(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("min_rigidity")
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("min_rigidity", "=", 3)],
        )
        g6_k3 = _g6(nx.complete_graph(3))
        g6_p4 = _g6(nx.path_graph(4))
        g6_diamond = _g6(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]))
        graphs = {r["graph"] for r in rows}
        # K3 is minimally d-rigid for all d >= n-1=2, so d=3 matches K3 (stored=-2 >= -3)
        assert g6_k3 in graphs
        # P4 (stored=1) and diamond (stored=2) are not minimally 3-rigid
        assert g6_p4 not in graphs
        assert g6_diamond not in graphs
