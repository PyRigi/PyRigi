"""
This file allows skipping `from pyrigi import Framework`
in section Examples in docstrings of `FrameworkBase`.
"""

import pytest

from pyrigi import Framework, Graph


@pytest.fixture(autouse=True)
def add_Framework(doctest_namespace):
    doctest_namespace["Framework"] = Framework
    doctest_namespace["Graph"] = Graph
