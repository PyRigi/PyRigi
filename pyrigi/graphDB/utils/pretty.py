"""Pretty-print helpers for query results."""
from __future__ import annotations

from typing import Any, Iterable, Optional, TextIO


def _normalize_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    return {"value": row}


def _stringify(value: Any, max_col_width: int) -> str:
    if value is None:
        text = "NULL"
    else:
        text = str(value)

    if max_col_width > 0 and len(text) > max_col_width:
        if max_col_width <= 3:
            return text[:max_col_width]
        return text[: max_col_width - 3] + "..."
    return text


def _collect_sample(rows: Iterable[Any], max_rows: Optional[int]) -> tuple[list[dict[str, Any]], bool]:
    sample: list[dict[str, Any]] = []
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be >= 0 or None")

    limit = None if max_rows is None else max_rows + 1
    for row in rows:
        sample.append(_normalize_row(row))
        if limit is not None and len(sample) >= limit:
            break

    truncated = max_rows is not None and len(sample) > max_rows
    if truncated:
        sample = sample[:max_rows]
    return sample, truncated


def format_result_table(
    rows: Iterable[Any],
    *,
    columns: Optional[list[str]] = None,
    max_rows: Optional[int] = 20,
    max_col_width: int = 48,
    show_index: bool = False,
) -> str:
    """Format query-like row data as an ASCII table.

    Parameters
    ----------
    rows:
        Iterable of row-like objects (dict rows are preferred).
    columns:
        Optional explicit column order.
    max_rows:
        Maximum number of rows to render. ``None`` prints all rows.
    max_col_width:
        Maximum width per rendered cell.
    show_index:
        If ``True``, include a leading ``#`` index column.
    """
    sample, truncated = _collect_sample(rows, max_rows=max_rows)
    if not sample:
        return "(no rows)"

    if columns is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in sample:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        columns = ordered

    headers = ["#"] + list(columns) if show_index else list(columns)

    table_rows: list[list[str]] = []
    for idx, row in enumerate(sample):
        values = [_stringify(row.get(col, ""), max_col_width=max_col_width) for col in columns]
        if show_index:
            values = [str(idx)] + values
        table_rows.append(values)

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _line(cells: list[str]) -> str:
        padded = [f" {cell:<{widths[i]}} " for i, cell in enumerate(cells)]
        return "|" + "|".join(padded) + "|"

    out = [sep, _line(headers), sep]
    for row in table_rows:
        out.append(_line(row))
    out.append(sep)

    if truncated:
        out.append("(more rows not shown; increase max_rows to view all)")

    return "\n".join(out)


def pretty_print_table(
    rows: Iterable[Any],
    *,
    columns: Optional[list[str]] = None,
    max_rows: Optional[int] = 20,
    max_col_width: int = 48,
    show_index: bool = False,
    file: Optional[TextIO] = None,
) -> str:
    """Format and print query-like row data as a table.

    Returns the printed string for convenience.
    """
    text = format_result_table(
        rows,
        columns=columns,
        max_rows=max_rows,
        max_col_width=max_col_width,
        show_index=show_index,
    )
    print(text, file=file)
    return text
