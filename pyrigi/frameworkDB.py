"""
This is a module for providing common types of frameworks.
"""

from pyrigi.framework import Framework
import pyrigi.graphDB as graphs
import pyrigi.misc as misc

import sympy as sp


def Cycle(n: int, d: int = 2) -> Framework:
    """Return d-dimensional framework of the n-cycle."""
    misc.check_integrality_and_range(n, "number of vertices n", 3)
    misc.check_integrality_and_range(d, "dimension d", 1)
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


def Square() -> Framework:
    """Framework of the 4-cycle with square realization in the plane"""
    return Framework(graphs.Cycle(4), {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


def Diamond() -> Framework:
    """Framework of the diamond with square realization in the plane"""
    return Framework(graphs.Diamond(), {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


def Complete(n: int, d: int = 2) -> Framework:
    """
    Return d-dimensional framework of the complete graph on n vertices.

    TODO
    ----
    Describe the generated realization.
    """
    misc.check_integrality_and_range(n, "number of vertices n", 1)
    misc.check_integrality_and_range(d, "dimension d", 1)
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


def Path(n: int, d: int = 2) -> Framework:
    """Return d-dimensional framework of the path graph on n vertices."""
    misc.check_integrality_and_range(n, "number of vertices n", 2)
    misc.check_integrality_and_range(d, "dimension d", 1)
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


def ThreePrism(realization: str = None) -> Framework:
    """
    Return 3-prism framework.

    Parameters
    ----------
    realization:
        If ``"parallel"``, a realization with the three edges that are not
        in any 3-cycle being parallel is returned.
        If ``"flexible"``, a continuously flexible realization is returned.
        Otherwise (default), a general realization is returned.
    """
    if realization == "parallel":
        return Framework(
            graphs.ThreePrism(),
            {0: (0, 0), 1: (2, 0), 2: (1, 2), 3: (0, 6), 4: (2, 6), 5: (1, 4)},
        )
    if realization == "flexible":
        return Framework(
            graphs.ThreePrism(),
            {0: (0, 0), 1: (2, 0), 2: (1, 2), 3: (0, 4), 4: (2, 4), 5: (1, 6)},
        )
    return Framework(
        graphs.ThreePrism(),
        {0: (0, 0), 1: (3, 0), 2: (2, 1), 3: (0, 4), 4: (2, 4), 5: (1, 3)},
    )


def ThreePrismPlusEdge() -> Framework:
    """Return a framework of the 3-prism graph with one extra edge."""
    G = ThreePrism()
    G.add_edge([0, 5])
    return G


def CompleteBipartite(m: int, n: int, realization: str = None) -> Framework:
    """
    Return a complete bipartite framework on m+n vertices in the plane.

    Parameters
    ----------
    realization:
        If ``"dixonI"``, a realization with one part on the x-axis and
        the other on the y-axis is returned.
        Otherwise (default), a "general" realization is returned.

    Todo
    ----
    Implement realization in higher dimensions.
    """
    misc.check_integrality_and_range(m, "size m", 1)
    misc.check_integrality_and_range(n, "size n", 1)
    if realization == "dixonI":
        return Framework(
            graphs.CompleteBipartite(m, n),
            {i: [0, (i + 1) * (-1) ** i] for i in range(m)}
            | {i: [(i - m + 1) * (-1) ** i, 0] for i in range(m, m + n)},
        )
    return Framework(
        graphs.CompleteBipartite(m, n),
        {
            i: [
                sp.cos(i * sp.pi / max([1, m - 1])),
                sp.sin(i * sp.pi / max([1, m - 1])),
            ]
            for i in range(m)
        }
        | {
            i: [
                1 + 2 * sp.cos((i - m) * sp.pi / max([1, n - 1])),
                3 + 2 * sp.sin((i - m) * sp.pi / max([1, n - 1])),
            ]
            for i in range(m, m + n)
        },
    )


def K33plusEdge() -> Framework:
    """Return a framework of the complete bipartite graph on 3+3 vertices plus an edge."""
    G = CompleteBipartite(3, 3, "dixonI")
    G.add_edge([0, 1])
    return G
