"""
This is a module for providing common types of frameworks.
"""

from pyrigi.framework import Framework
import pyrigi.graphDB as graphs
import pyrigi.misc as misc


def Cycle(n: int, d: int = 2):
    """Return d-dimensional framework of the n-cycle."""
    misc.check_integrality_and_range(n, "number of vertices n", 3)
    misc.check_integrality_and_range(d, "dimension", 1)
    if n - 1 <= d:
        return Framework.Simplicial(graphs.Cycle(n), d)
    elif d == 1:
        return Framework.Collinear(graphs.Cycle(n), 1)
    elif d == 2:
        return Framework.Circular(graphs.Cycle(n))
    raise ValueError(
        "The number of vertices n has to be at most d+1, or d must be 1 or 2 "
        f"(now (d, n) = ({d}, {n})."
    )


def Square():
    """Framework of the 4-cycle with square realization in the plane"""
    return Framework(graphs.Cycle(4), {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


def Diamond():
    """Framework of the diamond with square realization in the plane"""
    return Framework(graphs.Diamond(), {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


def Complete(n: int, d: int = 2):
    """Return d-dimensional framework of the complete graph on n vertices."""
    misc.check_integrality_and_range(n, "number of vertices n", 1)
    misc.check_integrality_and_range(d, "dimension", 1)
    if n - 1 <= d:
        return Framework.Simplicial(graphs.Complete(n), d)
    elif d == 1:
        return Framework.Collinear(graphs.Complete(n), 1)
    elif d == 2:
        return Framework.Circular(graphs.Complete(n))
    raise ValueError(
        "The number of vertices n has to be at most d+1, or d must be 1 or 2 "
        f"(now (d, n) = ({d}, {n})."
    )


def Path(n: int, d: int = 2):
    """Return d-dimensional framework of the path graph on n vertices."""
    misc.check_integrality_and_range(n, "number of vertices n", 2)
    misc.check_integrality_and_range(d, "dimension", 1)
    if n - 1 <= d:
        return Framework.Simplicial(graphs.Path(n), d)
    elif d == 1:
        return Framework.Collinear(graphs.Path(n), 1)
    elif d == 2:
        return Framework.Circular(graphs.Path(n))
    raise ValueError(
        "The number of vertices n has to be at most d+1, or d must be 1 or 2 "
        f"(now (d, n) = ({d}, {n})."
    )
