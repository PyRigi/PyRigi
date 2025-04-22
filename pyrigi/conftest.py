"""
This file allows skipping `from pyrigi import Graph`
in section Examples in docstrings.
"""

import pytest

from pyrigi import Graph


@pytest.fixture(autouse=True)
def add_Graph(doctest_namespace):
    doctest_namespace["Graph"] = Graph
