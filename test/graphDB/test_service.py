"""End-to-end tests for GraphStoreService."""

import pytest
import networkx as nx

from pyrigi.graphDB import AndExpr, GraphStoreService, OrExpr, QueryFilter


@pytest.fixture
def store():
    """In-memory store, initialised and ready to use."""
    s = GraphStoreService(":memory:").init()
    yield s
    s.close()


@pytest.fixture
def store_with_data(store, tmp_path):
    """Store with a small g6 file ingested (5-vertex connected graphs)."""
    g6_file = tmp_path / "graphs.g6"
    # Write a handful of known graph6 strings (n=5 connected graphs)
    graphs = [
        nx.complete_graph(5),
        nx.path_graph(5),
        nx.cycle_graph(5),
    ]
    lines = []
    for g in graphs:
        b = nx.to_graph6_bytes(g, header=False).strip()
        lines.append(b.decode("ascii"))
    g6_file.write_text("\n".join(lines) + "\n")
    store.ingest(str(g6_file))
    return store


class TestInit:
    def test_init_returns_self(self):
        s = GraphStoreService(":memory:")
        result = s.init()
        assert result is s
        s.close()

    def test_context_manager(self):
        with GraphStoreService(":memory:") as s:
            assert s.count() == 0

    def test_requires_init(self):
        s = GraphStoreService(":memory:")
        with pytest.raises(RuntimeError, match="init\\(\\)"):
            s.count()


class TestIngest:
    def test_ingest_counts_parse_errors(self, store, tmp_path):
        g6_file = tmp_path / "test.g6"
        g = nx.complete_graph(3)
        good = nx.to_graph6_bytes(g, header=False).strip().decode("ascii")
        g6_file.write_text(good + "\n" + "!!!INVALID!!!\n")
        stats = store.ingest(str(g6_file))
        assert stats.errors == 1
        assert stats.inserted == 1

    def test_ingest_mid_batch_flush(self, store, tmp_path):
        g6_file = tmp_path / "test.g6"
        lines = []
        for n in [3, 4, 5]:
            g = nx.complete_graph(n)
            lines.append(nx.to_graph6_bytes(g, header=False).strip().decode("ascii"))
        g6_file.write_text("\n".join(lines) + "\n")
        # batch_size=1 forces a flush after each graph, exercising the mid-loop path
        stats = store.ingest(str(g6_file), batch_size=1)
        assert stats.inserted == 3

    def test_ingest_from_file(self, store, tmp_path):
        g6_file = tmp_path / "test.g6"
        g = nx.complete_graph(3)
        g6 = nx.to_graph6_bytes(g, header=False).strip().decode("ascii")
        g6_file.write_text(g6 + "\n")
        stats = store.ingest(str(g6_file))
        assert stats.inserted == 1
        assert stats.skipped == 0
        assert store.count() == 1

    def test_ingest_idempotent(self, store, tmp_path):
        g6_file = tmp_path / "test.g6"
        g = nx.complete_graph(3)
        g6 = nx.to_graph6_bytes(g, header=False).strip().decode("ascii")
        g6_file.write_text(g6 + "\n")
        store.ingest(str(g6_file))
        stats2 = store.ingest(str(g6_file))
        assert stats2.inserted == 0
        assert stats2.skipped == 1

    def test_ingest_skips_single_vertex(self, store, tmp_path):
        # graph6 for a single vertex is "@"
        g6_file = tmp_path / "test.g6"
        g6_file.write_text("@\n")
        store.ingest(str(g6_file))
        assert store.count() == 0

    def test_ingest_from_directory(self, store, tmp_path):
        for i, n in enumerate([3, 4]):
            g = nx.complete_graph(n)
            g6 = nx.to_graph6_bytes(g, header=False).strip().decode("ascii")
            (tmp_path / f"graph_{i}.g6").write_text(g6 + "\n")
        stats = store.ingest(str(tmp_path))
        assert stats.inserted == 2
        assert stats.files_processed == 2


class TestAddColumn:
    def test_add_column_caches_populator(self, store_with_data):
        store_with_data.add_column(
            "cached_ec",
            "INTEGER",
            populator=lambda row: row["num_edges"],
        )
        # populate_column without explicit populator should use the cached one
        stats = store_with_data.populate_column("cached_ec")
        assert stats.errors == 0
        assert stats.processed == store_with_data.count()

    def test_add_custom_column(self, store):
        col = store.add_column("density", "REAL", description="Edge density")
        assert col.name == "density"
        names = [c.name for c in store.list_columns()]
        assert "density" in names

    def test_add_column_raises_for_default(self, store):
        with pytest.raises(ValueError, match="built-in default"):
            store.add_column("rigidity", "INTEGER")

    def test_add_column_idempotent(self, store):
        store.add_column("score", "REAL")
        store.add_column("score", "REAL")  # should not raise
        cols = [c.name for c in store.list_columns()]
        assert cols.count("score") == 1


class TestPopulateColumn:
    def test_populate_custom_column(self, store_with_data):
        store_with_data.add_column("edge_count_copy", "INTEGER")
        stats = store_with_data.populate_column(
            "edge_count_copy",
            populator=lambda row: row["num_edges"],
        )
        assert stats.errors == 0
        assert stats.processed == store_with_data.count()

    def test_populate_column_error_increments_stats(self, store_with_data):
        store_with_data.add_column("boom", "INTEGER")
        stats = store_with_data.populate_column(
            "boom",
            populator=lambda _: 1 / 0,  # noqa: U101  # always raises
        )
        assert stats.errors == store_with_data.count()
        assert stats.processed == 0

    def test_populate_raises_without_populator(self, store_with_data):
        store_with_data.add_column("no_pop", "INTEGER")
        with pytest.raises(RuntimeError, match="No populator"):
            store_with_data.populate_column("no_pop")

    def test_populate_raises_unknown_column(self, store):
        with pytest.raises(KeyError):
            store.populate_column("nonexistent_column")

    def test_count_unpopulated(self, store_with_data):
        store_with_data.add_column("unpop", "REAL")
        unpopulated = store_with_data.count_unpopulated("unpop")
        assert unpopulated == store_with_data.count()

    def test_count_unpopulated_unknown_column_raises(self, store):
        with pytest.raises(KeyError, match="Unknown column"):
            store.count_unpopulated("does_not_exist")

    def test_populate_column_batch_size(self, store_with_data):
        store_with_data.add_column("batch_test", "INTEGER")
        total = store_with_data.count()
        # batch_size=1 forces a flush after every single row, exercising both
        # the mid-loop flush and the tail flush paths
        stats = store_with_data.populate_column(
            "batch_test",
            populator=lambda row: row["num_edges"],
            batch_size=1,
        )
        assert stats.errors == 0
        assert stats.processed == total
        rows = store_with_data.fetch(select=["num_edges", "batch_test"])
        assert all(r["batch_test"] == r["num_edges"] for r in rows)

    def test_populate_all_rows_has_access_to_all_fields(self, store_with_data):
        store_with_data.add_column("density", "REAL")
        stats = store_with_data.populate_column(
            "density",
            all_rows=True,
            populator=lambda row: row["num_edges"]
            / (row["num_vertices"] * (row["num_vertices"] - 1) / 2),
        )
        assert stats.errors == 0
        assert stats.processed == store_with_data.count()


class TestFetch:
    def test_fetch_all(self, store_with_data):
        rows = store_with_data.fetch()
        assert len(rows) == 3

    def test_fetch_with_select(self, store_with_data):
        rows = store_with_data.fetch(select=["graph", "num_vertices"])
        assert all(set(r.keys()) == {"graph", "num_vertices"} for r in rows)

    def test_fetch_with_filter(self, store_with_data):
        rows = store_with_data.fetch(filters=[QueryFilter("num_vertices", "=", 5)])
        assert all(r["num_vertices"] == 5 for r in rows)

    def test_fetch_with_grouped_expr(self, store_with_data):
        rows = store_with_data.fetch(
            select=["num_edges"],
            expr=OrExpr(
                [
                    QueryFilter("num_edges", "=", 4),
                    QueryFilter("num_edges", "=", 5),
                ]
            ),
            order_by="num_edges",
        )
        assert [r["num_edges"] for r in rows] == [4, 5]

    def test_fetch_with_filters_and_expr_are_anded(self, store_with_data):
        rows = store_with_data.fetch(
            select=["num_edges"],
            filters=[QueryFilter("num_edges", ">", 4)],
            expr=OrExpr(
                [
                    QueryFilter("num_edges", "=", 4),
                    QueryFilter("num_edges", "=", 10),
                ]
            ),
        )
        assert len(rows) == 1
        assert rows[0]["num_edges"] == 10

    def test_fetch_with_offset(self, store_with_data):
        all_rows = store_with_data.fetch(
            select=["num_edges"], order_by="num_edges", asc=True
        )
        offset_rows = store_with_data.fetch(
            select=["num_edges"],
            order_by="num_edges",
            asc=True,
            limit=10,
            offset=1,
        )
        assert offset_rows == all_rows[1:]

    def test_fetch_with_limit(self, store_with_data):
        rows = store_with_data.fetch(limit=1)
        assert len(rows) == 1

    def test_fetch_order_by(self, store_with_data):
        rows = store_with_data.fetch(
            select=["num_edges"],
            order_by="num_edges",
            asc=True,
        )
        edges = [r["num_edges"] for r in rows]
        assert edges == sorted(edges)

    def test_fetch_order_by_desc(self, store_with_data):
        rows = store_with_data.fetch(
            select=["num_edges"],
            order_by="num_edges",
            asc=False,
        )
        edges = [r["num_edges"] for r in rows]
        assert edges == sorted(edges, reverse=True)

    def test_fetch_uses_runtime_fetch_strategy(self, store_with_data):
        # Runtime strategy should override default translation in this session.
        store_with_data.add_column(
            "score",
            "INTEGER",
            fetch_strategy=lambda col, _, val: (f"{col} >= ?", [val]),  # noqa: U101
        )
        store_with_data.populate_column("score", populator=lambda row: row["num_edges"])

        rows = store_with_data.fetch(
            select=["graph", "score"],
            filters=[QueryFilter("score", "=", 5)],
            order_by="score",
        )
        # Data has edge counts [10, 4, 5], so >= 5 should return two rows.
        assert len(rows) == 2
        assert all(r["score"] >= 5 for r in rows)

    def test_iter_fetch_matches_fetch(self, store_with_data):
        eager = store_with_data.fetch(
            select=["graph", "num_edges"],
            filters=[QueryFilter("num_vertices", "=", 5)],
            order_by="num_edges",
        )
        streamed = list(
            store_with_data.iter_fetch(
                select=["graph", "num_edges"],
                filters=[QueryFilter("num_vertices", "=", 5)],
                order_by="num_edges",
            )
        )
        assert streamed == eager

    def test_iter_fetch_with_networkx_mapper(self, store_with_data):
        graphs = list(
            store_with_data.iter_fetch(
                select=["graph"],
                filters=[QueryFilter("num_vertices", "=", 5)],
                mapper=lambda row: nx.from_graph6_bytes(row["graph"].encode("ascii")),
            )
        )
        assert len(graphs) == 3
        assert all(isinstance(g, nx.Graph) for g in graphs)

    def test_fetch_with_mapper_materializes_mapped_results(self, store_with_data):
        values = store_with_data.fetch(
            select=["num_edges"],
            order_by="num_edges",
            mapper=lambda row: row["num_edges"],
        )
        assert values == sorted(values)

    def test_service_format_results(self, store_with_data):
        rows = store_with_data.fetch(select=["num_vertices", "num_edges"], limit=2)
        text = store_with_data.format_results(rows, show_index=True)
        assert "num_vertices" in text
        assert "num_edges" in text
        assert "#" in text

    def test_service_pretty_print_results(self, store_with_data, capsys):
        rows = store_with_data.fetch(select=["num_vertices", "num_edges"], limit=2)
        text = store_with_data.pretty_print_results(rows, show_index=True)
        out = capsys.readouterr().out
        assert text in out
        assert "num_vertices" in out


class TestQueryBuilder:
    def test_fluent_query(self, store_with_data):
        rows = (
            store_with_data.query()
            .select(["graph", "num_vertices", "num_edges"])
            .where([QueryFilter("num_vertices", "=", 5)])
            .order_by("num_edges")
            .limit(10)
            .fetch()
        )
        assert all(r["num_vertices"] == 5 for r in rows)

    def test_filter_shorthand(self, store_with_data):
        rows = store_with_data.query().filter("num_vertices", "=", 5).fetch()
        assert len(rows) > 0

    def test_where_any_groups(self, store_with_data):
        rows = (
            store_with_data.query()
            .select(["num_edges"])
            .where_any(
                [
                    QueryFilter("num_edges", "=", 4),
                    QueryFilter("num_edges", "=", 5),
                ]
            )
            .where_any(
                [
                    QueryFilter("num_vertices", "=", 5),
                    QueryFilter("num_vertices", "=", 7),
                ]
            )
            .order_by("num_edges")
            .fetch()
        )
        assert [r["num_edges"] for r in rows] == [4, 5]

    def test_where_expr_nested_groups(self, store_with_data):
        rows = (
            store_with_data.query()
            .select(["num_edges"])
            .where_expr(
                AndExpr(
                    [
                        OrExpr(
                            [
                                QueryFilter("num_edges", "=", 4),
                                QueryFilter("num_edges", "=", 5),
                            ]
                        ),
                        OrExpr(
                            [
                                QueryFilter("num_vertices", "=", 5),
                                QueryFilter("num_vertices", "=", 7),
                            ]
                        ),
                    ]
                )
            )
            .order_by("num_edges")
            .fetch()
        )
        assert [r["num_edges"] for r in rows] == [4, 5]

    def test_iter_fetch_on_builder_with_mapper(self, store_with_data):
        rows = list(
            store_with_data.query()
            .select(["graph"])
            .filter("num_vertices", "=", 5)
            .iter_fetch(mapper=lambda row: row["graph"])
        )
        assert len(rows) == 3
        assert all(isinstance(v, str) for v in rows)

    def test_query_builder_format_results(self, store_with_data):
        text = (
            store_with_data.query()
            .select(["num_vertices", "num_edges"])
            .order_by("num_edges")
            .limit(2)
            .format_results()
        )
        assert "num_vertices" in text
        assert "num_edges" in text

    def test_query_builder_pretty_print(self, store_with_data, capsys):
        text = (
            store_with_data.query()
            .select(["num_vertices", "num_edges"])
            .order_by("num_edges")
            .limit(2)
            .pretty_print()
        )
        out = capsys.readouterr().out
        assert text in out
        assert "num_vertices" in out


class TestInfo:
    def test_info_structure(self, store):
        info = store.info()
        assert "total_graphs" in info
        assert "columns" in info
        assert any(c["name"] == "rigidity" for c in info["columns"])


class TestUpdateColumnPopulator:
    def test_update_column_populator(self, store_with_data):
        store_with_data.update_column_populator(
            "rigidity",
            populator=lambda _: 2,  # noqa: U101
        )
        stats = store_with_data.populate_column("rigidity")
        assert stats.processed == store_with_data.count()
        rows = store_with_data.fetch(select=["rigidity"])
        assert all(r["rigidity"] == 2 for r in rows)

    def test_update_column_populator_preserves_existing_ref(self, store):
        before = store.get_column("rigidity")
        assert before is not None
        assert before.populator_ref is not None

        store.update_column_populator("rigidity", populator=lambda _: 2)  # noqa: U101

        after = store.get_column("rigidity")
        assert after is not None
        assert after.populator_ref == before.populator_ref

    def test_update_column_populator_unknown_column_raises(self, store):
        with pytest.raises(KeyError):
            store.update_column_populator("nonexistent_col")

    def test_update_column_populator_clears_cache(self, store_with_data):
        # Prime the cache by populating with an explicit lambda
        store_with_data.update_column_populator(
            "rigidity", populator=lambda _: 1  # noqa: U101
        )
        # Now clear the runtime populator — the cache entry should be removed
        store_with_data.update_column_populator("rigidity", populator=None)
        # populate_column must now fall back to the populator_ref, not the cache
        stats = store_with_data.populate_column("rigidity")
        assert stats.errors == 0

    def test_update_column_populator_can_clear_ref_explicitly(self, store):
        store.update_column_populator("rigidity", populator_ref=None)
        col = store.get_column("rigidity")
        assert col is not None
        assert col.populator_ref is None


class TestDropColumn:
    def test_drop_column_removes_from_registry(self, store):
        store.add_column("tmp_col", "INTEGER")
        assert store.get_column("tmp_col") is not None
        store.drop_column("tmp_col")
        assert store.get_column("tmp_col") is None

    def test_drop_column_removes_from_schema(self, store):
        store.add_column("tmp_col", "INTEGER")
        store.drop_column("tmp_col")
        cols = store._db._existing_columns()
        assert "tmp_col" not in cols

    def test_drop_column_clears_caches(self, store):
        fn = lambda row: 0  # noqa: E731
        store.add_column("tmp_col", "INTEGER", populator=fn, fetch_strategy=fn)
        assert "tmp_col" in store._populator_cache
        assert "tmp_col" in store._fetch_cache
        store.drop_column("tmp_col")
        assert "tmp_col" not in store._populator_cache
        assert "tmp_col" not in store._fetch_cache

    def test_drop_column_raises_for_default(self, store):
        with pytest.raises(ValueError, match="built-in"):
            store.drop_column("rigidity")

    def test_drop_column_raises_for_unknown(self, store):
        with pytest.raises(KeyError):
            store.drop_column("nonexistent_col")


class TestDeleteGraph:
    def test_delete_graph_returns_true_when_found(self, store_with_data):
        g6 = nx.to_graph6_bytes(nx.complete_graph(5), header=False).strip().decode()
        assert store_with_data.delete_graph(g6) is True

    def test_delete_graph_returns_false_when_not_found(self, store_with_data):
        assert store_with_data.delete_graph("not_a_real_graph6") is False

    def test_delete_graph_reduces_count(self, store_with_data):
        before = store_with_data.count()
        g6 = nx.to_graph6_bytes(nx.path_graph(5), header=False).strip().decode()
        store_with_data.delete_graph(g6)
        assert store_with_data.count() == before - 1

    def test_delete_where_by_filter(self, store_with_data):
        before = store_with_data.count()
        deleted = store_with_data.delete_where(
            filters=[QueryFilter("num_vertices", "=", 5)]
        )
        assert deleted == before
        assert store_with_data.count() == 0

    def test_delete_where_returns_rowcount(self, store_with_data):
        deleted = store_with_data.delete_where(
            filters=[QueryFilter("num_edges", "<", 0)]
        )
        assert deleted == 0

    def test_delete_where_no_filters_deletes_all(self, store_with_data):
        assert store_with_data.count() > 0
        store_with_data.delete_where()
        assert store_with_data.count() == 0


class TestUpdateColumnFetchStrategy:
    def test_update_column_fetch_strategy_caches_callable(self, store):
        fn = lambda col, op, val: (f"{col} = ?", [val])  # noqa: E731
        store.update_column_fetch_strategy("rigidity", fetch_strategy=fn)
        assert store._fetch_cache.get("rigidity") is fn

    def test_update_column_fetch_strategy_clears_cache_when_none(self, store):
        fn = lambda col, op, val: (f"{col} = ?", [val])  # noqa: E731
        store.update_column_fetch_strategy("rigidity", fetch_strategy=fn)
        store.update_column_fetch_strategy("rigidity", fetch_strategy=None)
        assert "rigidity" not in store._fetch_cache

    def test_update_column_fetch_strategy_persists_ref(self, store):
        ref = "pyrigi.graphDB.defaults.fetch_strategies:_rigidity_fetch_strategy"
        store.update_column_fetch_strategy("rigidity", fetch_ref=ref)
        col = store.get_column("rigidity")
        assert col is not None
        assert col.fetch_ref == ref

    def test_update_column_fetch_strategy_preserves_existing_ref_when_unset(self, store):
        before = store.get_column("rigidity")
        assert before is not None
        original_ref = before.fetch_ref
        store.update_column_fetch_strategy("rigidity", fetch_strategy=lambda c, o, v: (c, []))
        after = store.get_column("rigidity")
        assert after is not None
        assert after.fetch_ref == original_ref

    def test_update_column_fetch_strategy_unknown_column_raises(self, store):
        with pytest.raises(KeyError):
            store.update_column_fetch_strategy("nonexistent_col")
