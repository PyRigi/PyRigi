"""
pyrigi.graphDB.constants.operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SQL operator whitelist used by :class:`~pyrigi.graphDB.models.filters.QueryFilter`.
"""

VALID_OPERATORS = frozenset(
    {"=", "!=", "<", "<=", ">", ">=", "IN", "BETWEEN", "LIKE", "IS NULL", "IS NOT NULL"}
)
