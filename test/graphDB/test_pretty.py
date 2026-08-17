"""Tests for pretty-print utilities."""

import pytest

from pyrigi.graphDB.utils.pretty import format_result_table, pretty_print_table


class TestPrettyUtils:
    def test_format_result_table_basic(self):
        text = format_result_table(
            [
                {"a": 1, "b": "x"},
                {"a": 2, "b": "y"},
            ]
        )
        assert "a" in text
        assert "b" in text
        assert "1" in text
        assert "2" in text

    def test_format_result_table_truncation_notice(self):
        text = format_result_table(
            [
                {"a": 1},
                {"a": 2},
            ],
            max_rows=1,
        )
        assert "more rows not shown" in text

    def test_format_result_table_non_dict_rows(self):
        text = format_result_table([1, 2, 3], max_rows=2)
        assert "value" in text
        assert "1" in text
        assert "2" in text

    def test_empty_rows_returns_no_rows_sentinel(self):
        assert format_result_table([]) == "(no rows)"

    def test_none_value_renders_as_null(self):
        text = format_result_table([{"col": None}])
        assert "NULL" in text

    def test_long_value_gets_ellipsis_truncation(self):
        text = format_result_table([{"col": "x" * 100}], max_col_width=10)
        assert "..." in text

    def test_very_short_max_col_width_no_ellipsis(self):
        # max_col_width <= 3 slices without appending "..."
        text = format_result_table([{"col": "abcdef"}], max_col_width=2)
        assert "..." not in text

    def test_negative_max_rows_raises(self):
        with pytest.raises(ValueError, match="max_rows"):
            format_result_table([{"a": 1}], max_rows=-1)

    def test_row_as_object_with_dict_uses_vars(self):
        class Obj:
            def __init__(self):
                self.score = 7

        text = format_result_table([Obj()])
        assert "score" in text
        assert "7" in text

    def test_pretty_print_table_prints_and_returns(self, capsys):
        text = pretty_print_table([{"x": 1}])
        out = capsys.readouterr().out
        assert text in out
        assert "x" in out
