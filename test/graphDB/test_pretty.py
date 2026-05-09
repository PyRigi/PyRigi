"""Tests for pretty-print utilities."""

from pyrigi.graphDB.utils.pretty import format_result_table


class TestPrettyUtils:
    def test_format_result_table_basic(self):
        text = format_result_table([
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
        ])
        assert "a" in text
        assert "b" in text
        assert "1" in text
        assert "2" in text

    def test_format_result_table_truncation_notice(self):
        text = format_result_table([
            {"a": 1},
            {"a": 2},
        ], max_rows=1)
        assert "more rows not shown" in text

    def test_format_result_table_non_dict_rows(self):
        text = format_result_table([1, 2, 3], max_rows=2)
        assert "value" in text
        assert "1" in text
        assert "2" in text
