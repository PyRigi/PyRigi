"""
pyrigi.graphDB.constants.identifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared compiled regex for validating SQL identifiers (column names).
"""

import re

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
