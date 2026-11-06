"""
This file provides fixtures applied to both the tests and the doctests.
"""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    """
    Close all `matplotlib` figures after each test.
    """
    yield
    plt.close("all")
