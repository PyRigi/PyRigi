"""
This file allows skipping `from pyrigi import Framework`
in section Examples in docstrings of `FrameworkBase`.
"""

import pytest

from pyrigi import Framework


@pytest.fixture(autouse=True)
def add_Framework(doctest_namespace):
    doctest_namespace["Framework"] = Framework
