"""
pyrigi.graphDB.models.resolvers
~~~~~~~~~~~~~~~~~~~~~~~~~~
Private helpers for resolving importable callable references and
providing the default SQL fetch strategy.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


def _import_callable(ref: str) -> Callable:
    """Resolve an importable reference of the form ``"package.module:func"``."""
    if ":" not in ref:
        raise ValueError(
            f"Invalid callable reference {ref!r}. "
            "Expected format: 'package.module:function_name'"
        )
    module_path, attr = ref.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, attr)
    if not callable(obj):
        raise TypeError(f"{ref!r} resolved to a non-callable: {type(obj)}")
    return obj


def _default_fetch_strategy(column: str, operator: str, value: Any) -> tuple[str, list]:
    """Pass-through fetch strategy used when no custom one is registered."""
    op = operator.upper()
    if op == "IS NULL":
        return f"{column} IS NULL", []
    if op == "IS NOT NULL":
        return f"{column} IS NOT NULL", []
    if op == "IN":
        placeholders = ", ".join("?" * len(value))
        return f"{column} IN ({placeholders})", list(value)
    if op == "BETWEEN":
        lo, hi = value
        return f"{column} BETWEEN ? AND ?", [lo, hi]
    # default: binary operator
    return f"{column} {op} ?", [value]
