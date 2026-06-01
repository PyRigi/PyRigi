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
    def test_eq_includes_complete(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "=", 3)
        assert sql == "(rigidity = ? OR rigidity = -1)"
        assert params == [3]

    def test_eq_minus_one_is_complete_only(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "=", -1)
        assert sql == "(rigidity = ? OR rigidity = -1)"
        assert params == [-1]

    def test_in_includes_complete(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "IN", [1, 2])
        assert "IN (?, ?)" in sql
        assert "rigidity = -1" in sql
        assert params == [1, 2]

    def test_is_null_passthrough(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "IS NULL", None)
        assert sql == "rigidity IS NULL"
        assert params == []

    def test_is_not_null_passthrough(self):
        sql, params = _rigidity_fetch_strategy("rigidity", "IS NOT NULL", None)
        assert sql == "rigidity IS NOT NULL"
        assert params == []

    def test_operator_case_insensitive(self):
        sql, params = _rigidity_fetch_strategy("global_rigidity", "=", 2)
        assert "= -1" in sql
        assert params == [2]


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

    def test_in_expands_for_complete_graphs(self):
        sql, params = _min_rigidity_fetch_strategy("min_rigidity", "IN", [1, 2])
        assert "IN (?, ?)" in sql
        assert "min_rigidity < 0" in sql
        assert "min_rigidity >= ?" in sql
        assert params == [1, 2, -2]

    def test_in_uses_max_value_as_bound(self):
        _, params = _min_rigidity_fetch_strategy("min_rigidity", "IN", [1, 3])
        assert params == [1, 3, -3]

    def test_is_null_passthrough(self):
        sql, params = _min_rigidity_fetch_strategy("min_rigidity", "IS NULL", None)
        assert sql == "min_rigidity IS NULL"
        assert params == []

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
    def test_eq_minus_one_returns_complete_graph(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("rigidity")
        # K3 is stored as -1 (complete-graph sentinel); query via = -1
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("rigidity", "=", -1)],
        )
        g6_k3 = _g6(nx.complete_graph(3))
        assert any(r["graph"] == g6_k3 for r in rows)

    def test_gte_rejected_for_rigidity_column(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        with pytest.raises(ValueError, match="not supported"):
            store.fetch(filters=[QueryFilter("rigidity", ">=", 1)])

    def test_eq_d_returns_complete_graph(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("rigidity")
        # "= d" expands to (rigidity = d OR rigidity = -1), so K3 must appear
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("rigidity", "=", 1)],
        )
        g6_k3 = _g6(nx.complete_graph(3))
        assert any(r["graph"] == g6_k3 for r in rows)

    def test_in_returns_complete_graph(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("rigidity")
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("rigidity", "IN", [1, 2])],
        )
        g6_k3 = _g6(nx.complete_graph(3))
        assert any(r["graph"] == g6_k3 for r in rows)


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

    def test_min_rigidity_in_returns_complete_graph(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("min_rigidity")
        # IN [2] is equivalent to = 2 semantically: K3 (stored=-2) satisfies -2 >= -2
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("min_rigidity", "IN", [2])],
        )
        g6_k3 = _g6(nx.complete_graph(3))
        graphs = {r["graph"] for r in rows}
        assert g6_k3 in graphs

    def test_min_rigidity_in_multiple_values(self, store_with_rigidity_data):
        store = store_with_rigidity_data
        store.populate_column("min_rigidity")
        # IN [1, 2]: P4 (stored=1), diamond (stored=2), K3 (stored=-2, -2 >= -2) all match
        rows = store.fetch(
            select=["graph"],
            filters=[QueryFilter("min_rigidity", "IN", [1, 2])],
        )
        g6_p4 = _g6(nx.path_graph(4))
        g6_diamond = _g6(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]))
        g6_k3 = _g6(nx.complete_graph(3))
        graphs = {r["graph"] for r in rows}
        assert g6_p4 in graphs
        assert g6_diamond in graphs
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
