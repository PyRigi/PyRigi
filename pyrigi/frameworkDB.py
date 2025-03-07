"""
This is a module for providing common types of frameworks.
"""

import sympy as sp

import pyrigi._input_check as _input_check
import pyrigi.graphDB as graphs
from pyrigi.framework import Framework


def Cycle(n: int, dim: int = 2) -> Framework:
    """
    Return ``dim``-dimensional framework of the ``n``-cycle.

    The resulting realization depends on the dimension ``dim``
    and on the number ``n``:

    - If ``n-1<=dim``, then the cycle is realized as the vertices of the
      `dim`-dimensional simplex.
    - For ``dim==1``, the cycle is injectively realized on the 1-dimensional
      linear space generated by the vector ``[1,0,...,0]``.
    - If ``dim==2``, then the cycle is realized as the vertices of a regular polygon.
    """
    _input_check.integrality_and_range(n, "number of vertices n", 3)
    _input_check.integrality_and_range(dim, "dimension d", 1)

    if n - 1 <= dim:
        return Framework.Simplicial(graphs.Cycle(n), dim)
    elif dim == 1:
        return Framework.Collinear(graphs.Cycle(n), 1)
    elif dim == 2:
        return Framework.Circular(graphs.Cycle(n))
    raise ValueError(
        "The number of vertices n has to be at most d+1, or d must be 1 or 2 "
        f"(now (d, n) = ({dim}, {n})."
    )


def Square() -> Framework:
    """Return the 4-cycle with square realization in the plane."""
    return Framework(graphs.Cycle(4), {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


def Diamond() -> Framework:
    """Return the diamond with square realization in the plane."""
    return Framework(graphs.Diamond(), {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]})


def Cube() -> Framework:
    r"""Return the regular cube in $\RR^3$."""
    F = Framework(
        graphs.CubeWithDiagonal(),
        {
            0: [0, 0, 0],
            1: [1, 0, 0],
            2: [1, 1, 0],
            3: [0, 1, 0],
            4: [0, 0, 1],
            5: [1, 0, 1],
            6: [1, 1, 1],
            7: [0, 1, 1],
        },
    )
    F.delete_edge([0, 6])
    return F


def Octahedron(realization: str = "regular") -> Framework:
    r"""
    Return a framework of the octahedron in $\RR^3$.

    Parameters
    ----------
    realization:
        If ``"regular"``, a realization of the regular octahedron in $\RR^3$
        is returned.
        If ``"Bricard_line"``, a flexible Bricard's octahedron
        line-symmetric w.r.t. z-axis is returned.
        If ``"Bricard_plane"``, a flexible Bricard's octahedron
        plane-symmetric w.r.t. yz-plane is returned.
    """
    if realization == "regular":
        return Framework(
            graphs.Octahedral(),
            {
                0: [0, 0, "-sqrt(2)"],
                1: [0, 0, "sqrt(2)"],
                2: [-1, -1, 0],
                3: [1, 1, 0],
                4: [1, -1, 0],
                5: [-1, 1, 0],
            },
        )
    elif realization == "Bricard_line":
        a = 1
        b, c, d = 1, 2, 1
        e, f, g = 2, -1, -1
        # We define a realization symmetric w.r.t. z-axis.
        return Framework(
            graphs.Octahedral(),
            {
                0: [a, 0, 0],
                1: [-a, 0, 0],
                2: [b, c, d],
                3: [-b, -c, d],
                4: [e, f, g],
                5: [-e, -f, g],
            },
        )
    elif realization == "Bricard_plane":
        a, b = 1, 1
        c, d = 1, -2
        e = 1
        f, g = 1, 1
        # We define a realization symmetric w.r.t. yz-plane, which contains 0 and 1.
        return Framework(
            graphs.Octahedral(),
            {
                0: [0, a, b],
                1: [0, c, d],
                2: [e, 0, 0],
                3: [-e, 0, 0],
                4: [f, g, 0],
                5: [-f, g, 0],
            },
        )
    raise ValueError(f"The parameter `realization` cannot be {realization}.")


def Icosahedron() -> Framework:
    r"""Return the regular icosahedron in $\RR^3$."""
    phi = sp.sympify("(1+sqrt(5))/2")
    F = Framework(
        graphs.Icosahedral(),
        {
            0: (phi, 1, 0),
            1: (1, 0, phi),
            2: (-1, 0, phi),
            9: (-phi, 1, 0),
            4: (0, -phi, -1),
            5: (phi, -1, 0),
            6: (0, -phi, 1),
            7: (0, phi, -1),
            8: (0, phi, 1),
            3: (-phi, -1, 0),
            10: (-1, 0, -phi),
            11: (1, 0, -phi),
        },
    )
    return F


def Dodecahedron() -> Framework:
    r"""Return the regular dodecahedron in $\RR^3$."""
    phi = sp.sympify("(1+sqrt(5))/2")
    F = Framework(
        graphs.Dodecahedral(),
        {
            0: (1, 1, 1),
            1: (-1, 1, 1),
            2: (1, -1, 1),
            3: (1, 1, -1),
            4: (-1, -1, 1),
            5: (-1, 1, -1),
            6: (1, -1, -1),
            7: (-1, -1, -1),
            8: (0, phi, 1 / phi),
            9: (0, phi, -1 / phi),
            10: (0, -phi, 1 / phi),
            11: (0, -phi, -1 / phi),
            12: (1 / phi, 0, phi),
            13: (-1 / phi, 0, phi),
            14: (1 / phi, 0, -phi),
            15: (-1 / phi, 0, -phi),
            16: (phi, 1 / phi, 0),
            17: (phi, -1 / phi, 0),
            18: (-phi, 1 / phi, 0),
            19: (-phi, -1 / phi, 0),
        },
    )
    return F


def K33plusEdge() -> Framework:
    r"""
    Return the complete bipartite graph on 3+3 vertices plus an edge realized in $\RR^2$.
    """
    return Framework(
        graphs.K33plusEdge(),
        {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 0], 4: [1, 1], 5: [1, 2]},
    )


def Complete(n: int, dim: int = 2) -> Framework:
    """
    Return ``dim``-dimensional framework of the complete graph on ``n`` vertices.

    The resulting realization depends on the dimension ``dim``
    and on the number ``n``:

    - If ``n-1<=dim``, then the complete graph is realized on the vertices of the
      ``dim``-dimensional simplex. The complete graph is either given by the
      1-skeleton of the simplex itself when ``dim+1==n`` or else as the 1-skeleton
      of a facet.
    - For ``dim==1``, the complete graph is injectively realized on the
      1-dimensional linear space generated by the vector ``[1,0,...,0]``.
    - If ``dim==2``, then the complete graph is realized as the vertices
      of a regular polygon.
    """
    _input_check.integrality_and_range(n, "number of vertices n", 1)
    _input_check.integrality_and_range(dim, "dimension d", 1)
    if n - 1 <= dim:
        return Framework.Simplicial(graphs.Complete(n), dim)
    elif dim == 1:
        return Framework.Collinear(graphs.Complete(n), 1)
    elif dim == 2:
        return Framework.Circular(graphs.Complete(n))
    raise ValueError(
        "The number of vertices n has to be at most d+1, or d must be 1 or 2 "
        f"(now (d, n) = ({dim}, {n})."
    )


def Path(n: int, dim: int = 2) -> Framework:
    """
    Return ``dim``-dimensional framework of the path graph on ``n`` vertices.

    The resulting realization depends on the dimension ``dim`` and ``n``:

    - If ``n-1<=dim``, then the path is realized as the vertices of the
      `dim`-dimensional simplex.
    - For ``dim==1``, the path is injectively realized on the 1-dimensional linear space
      generated by the vector ``[1,0,...,0]``.
    - If ``dim==2``, then the path is realized as the vertices of a regular polygon.
    """
    _input_check.integrality_and_range(n, "number of vertices n", 2)
    _input_check.integrality_and_range(dim, "dimension d", 1)

    if n - 1 <= dim:
        return Framework.Simplicial(graphs.Path(n), dim)
    elif dim == 1:
        return Framework.Collinear(graphs.Path(n), 1)
    elif dim == 2:
        return Framework.Circular(graphs.Path(n))
    raise ValueError(
        "The number of vertices n has to be at most d+1, or d must be 1 or 2 "
        f"(now (d, n) = ({dim}, {n})."
    )


def ThreePrism(realization: str = None) -> Framework:
    """
    Return a framework of the 3-prism graph in the plane.

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
    """Return a framework of the 3-prism graph with one extra edge in the plane."""
    G = ThreePrism()
    G.add_edge([0, 5])
    return G


def CompleteBipartite(n1: int, n2: int, realization: str = None) -> Framework:
    """
    Return a complete bipartite framework on ``n1+n2`` vertices in the plane.

    Parameters
    ----------
    n1,n2:
        The sizes of the two parts.
    realization:
        If ``"dixonI"``, a realization with one part on the x-axis and
        the other on the y-axis is returned.
        Otherwise (default), a "general" realization is returned.

    Suggested Improvements
    ----------------------
    Implement realization in higher dimensions.
    """
    _input_check.integrality_and_range(n1, "size n1", 1)
    _input_check.integrality_and_range(n2, "size n2", 1)
    if realization == "dixonI":
        return Framework(
            graphs.CompleteBipartite(n1, n2),
            {i: [0, (i + 1) * (-1) ** i] for i in range(n1)}
            | {i: [(i - n1 + 1) * (-1) ** i, 0] for i in range(n1, n1 + n2)},
        )
    elif realization == "collinear":
        return Framework(
            graphs.CompleteBipartite(n1, n2), {i: [i, 0] for i in range(n1 + n2)}
        )
    return Framework(
        graphs.CompleteBipartite(n1, n2),
        {
            i: [
                sp.cos(i * sp.pi / max([1, n1 - 1])),
                sp.sin(i * sp.pi / max([1, n1 - 1])),
            ]
            for i in range(n1)
        }
        | {
            i: [
                1 + 2 * sp.cos((i - n1) * sp.pi / max([1, n2 - 1])),
                3 + 2 * sp.sin((i - n1) * sp.pi / max([1, n2 - 1])),
            ]
            for i in range(n1, n1 + n2)
        },
    )


def Frustum(n: int) -> Framework:
    """
    Return the $n$-Frustum with ``2*n`` vertices in dimension 2.

    Definitions
    -----------
    * :prf:ref:`n-Frustum <def-n-frustum>`
    """
    if n <= 2 or not isinstance(n, int):
        raise ValueError("`n` needs to be at least 3 but is {n}")
    realization = {
        j: (sp.cos(2 * j * sp.pi / n), sp.sin(2 * j * sp.pi / n)) for j in range(0, n)
    }
    realization.update(
        {
            (j + n): (2 * sp.cos(2 * j * sp.pi / n), 2 * sp.sin(2 * j * sp.pi / n))
            for j in range(0, n)
        }
    )
    F = Framework(graphs.Frustum(n), realization)
    return F


def CnSymmetricFourRegular(n: int = 8) -> Framework:
    """
    Return a $C_n$-symmetric framework in the plane.

    Definitions
    -----------
    * :prf:ref:`Example with a free group action <def-Cn-symmetric>`
    """
    if not n % 2 == 0 or n < 8:
        raise ValueError(
            "To generate this framework, the cyclic group "
            + "must have an even order of at least 8!"
        )
    return Framework(
        graphs.CnSymmetricFourRegular(n),
        {
            i: [
                sp.cos(2 * i * sp.pi / n),
                sp.sin(2 * i * sp.pi / n),
            ]
            for i in range(n)
        },
    )


def CnSymmetricFourRegularWithFixedVertex(n: int = 8) -> Framework:
    """
    Return a $C_n$-symmetric framework with a fixed vertex in the plane.

    The value ``n`` must be even and at least 8.

    The returned graph satisfies the expected symmetry-adapted Laman
    count for rotation but is infinitesimally flexible.

    Definitions
    -----------
    * :prf:ref:`Example with joint at origin <def-Cn-symmetric-joint-at-origin>`
    """
    if not n % 2 == 0 or n < 8:
        raise ValueError(
            "To generate this framework, the cyclic group "
            + "must have an even order of at least 8!"
        )
    return Framework(
        graphs.CnSymmetricFourRegularWithFixedVertex(n),
        {
            i: [
                sp.cos(2 * i * sp.pi / n),
                sp.sin(2 * i * sp.pi / n),
            ]
            for i in range(n)
        }
        | {
            i
            + n: [
                sp.Rational(9, 5) * sp.cos((2 * i) * sp.pi / n)
                - sp.sin((2 * i) * sp.pi / n),
                sp.Rational(9, 5) * sp.sin((2 * i) * sp.pi / n)
                + sp.cos((2 * i) * sp.pi / n),
            ]
            for i in range(n)
        }
        | {2 * n: (0, 0)},
    )


def ConnellyExampleSecondOrderRigidity() -> Framework:
    """
    Return an example by Connelly.

    This is an example of a (non-injective) 3-dimensional framework with a
    two-dimensional space of flexes and stresses. It appears in
    {{references}} {cite:p}`Connelly1996`. This framework is
    second-order rigid but not prestress stable.
    """
    F = Framework.from_points(
        [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)]
    )
    F.add_edges(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 5),
            (0, 6),
            (1, 2),
            (1, 4),
            (1, 6),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
            (4, 6),
            (5, 6),
        ]
    )
    return F
