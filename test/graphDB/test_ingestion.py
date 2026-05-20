"""Tests for pyrigi.graphDB.ingestion."""

import pytest
import networkx as nx

from pyrigi.graphDB.ingestion import DefaultColumnComputer, G6Reader, GraphParser


# ---------------------------------------------------------------------------
# G6Reader
# ---------------------------------------------------------------------------


class TestG6Reader:
    def test_reads_single_file(self, tmp_path):
        f = tmp_path / "test.g6"
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

    def test_reads_gz_file(self, tmp_path):
        import gzip

        gz_file = tmp_path / "test.g6.gz"
        with gzip.open(gz_file, "wt", encoding="ascii") as fh:
            fh.write("B?\nBw\n")
        strings = list(G6Reader(gz_file).iter_strings())
        assert "B?" in strings
        assert "Bw" in strings

    def test_iter_strings_with_file_yields_path_pairs(self, tmp_path):
        f = tmp_path / "test.g6"
        f.write_text("B?\nBw\n")
        pairs = list(G6Reader(f).iter_strings_with_file())
        assert len(pairs) == 2
        assert all(path == f for path, _ in pairs)
        assert {g6 for _, g6 in pairs} == {"B?", "Bw"}

    def test_empty_directory_logs_warning(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="pyrigi.graphDB.ingestion.reader"):
            files = G6Reader(tmp_path).files()
        assert files == []
        assert "No .g6" in caplog.text

    def test_unreadable_file_logs_error_and_yields_nothing(self, tmp_path, caplog):
        import logging

        bad = tmp_path / "bad.g6"
        bad.write_bytes(b"\xff\xfe")  # invalid ASCII — triggers decode error
        with caplog.at_level(logging.ERROR, logger="pyrigi.graphDB.ingestion.reader"):
            strings = list(G6Reader(bad).iter_strings())
        assert strings == []
        assert "Failed to read" in caplog.text


# ---------------------------------------------------------------------------
# GraphParser
# ---------------------------------------------------------------------------


class TestGraphParser:
    def test_parse_k3(self):
        parser = GraphParser()
        g = parser.parse("Bw")  # K_3
        assert g is not None
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 3

    def test_parse_path_2(self):
        parser = GraphParser()
        g = parser.parse("A_")  # K_2
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
