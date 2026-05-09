"""Tests for graph_db.ingestion."""
import io
import textwrap
from pathlib import Path

import pytest
import networkx as nx

from pyrigi.graphDB.ingestion import DefaultColumnComputer, G6Reader, GraphParser


# ---------------------------------------------------------------------------
# G6Reader
# ---------------------------------------------------------------------------

class TestG6Reader:
    def test_reads_single_file(self, tmp_path):
        f = tmp_path / "test.g6"
        # K3 in graph6
        f.write_text("B?\n")
        reader = G6Reader(f)
        strings = list(reader.iter_strings())
        assert "B?" in strings

    def test_reads_directory(self, tmp_path):
        (tmp_path / "a.g6").write_text("B?\n")
        (tmp_path / "b.g6").write_text("Bw\n")
        reader = G6Reader(tmp_path)
        strings = list(reader.iter_strings())
        assert len(strings) == 2

    def test_skips_comment_lines(self, tmp_path):
        f = tmp_path / "test.g6"
        f.write_text(">>graph6<<\nB?\n")
        strings = list(G6Reader(f).iter_strings())
        assert ">>graph6<<" not in strings
        assert "B?" in strings

    def test_raises_on_missing_source(self):
        with pytest.raises(FileNotFoundError):
            G6Reader("/nonexistent/path").files()

    def test_skips_non_g6_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not a graph\n")
        (tmp_path / "graph.g6").write_text("B?\n")
        files = G6Reader(tmp_path).files()
        assert all(str(f).endswith(".g6") or str(f).endswith(".g6.gz") for f in files)


# ---------------------------------------------------------------------------
# GraphParser
# ---------------------------------------------------------------------------

class TestGraphParser:
    def test_parse_k3(self):
        parser = GraphParser()
        g = parser.parse("Bw")   # K_3
        assert g is not None
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 3

    def test_parse_path_2(self):
        parser = GraphParser()
        g = parser.parse("A_")   # K_2
        assert g is not None
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 1

    def test_parse_invalid_returns_none(self):
        parser = GraphParser(strict=False)
        g = parser.parse("!!!INVALID!!!")
        assert g is None

    def test_parse_invalid_strict_raises(self):
        parser = GraphParser(strict=True)
        with pytest.raises(Exception):
            parser.parse("!!!INVALID!!!")


# ---------------------------------------------------------------------------
# DefaultColumnComputer
# ---------------------------------------------------------------------------

class TestDefaultColumnComputer:
    def test_k3(self):
        g = nx.complete_graph(3)
        g6 = "Bw"
        row = DefaultColumnComputer().compute(g6, g)
        assert row["graph"] == g6
        assert row["num_vertices"] == 3
        assert row["num_edges"] == 3
        assert row["min_degree"] == 2
        assert row["max_degree"] == 2

    def test_path_graph(self):
        g = nx.path_graph(4)
        # Get g6 string from networkx
        g6_bytes = nx.to_graph6_bytes(g, header=False).strip()
        g6 = g6_bytes.decode("ascii")
        row = DefaultColumnComputer().compute(g6, g)
        assert row["num_vertices"] == 4
        assert row["num_edges"] == 3
        assert row["min_degree"] == 1  # endpoints
        assert row["max_degree"] == 2  # middle nodes
