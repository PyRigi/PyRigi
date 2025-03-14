import pytest
import sympy as sp

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi.misc import is_zero


def test_Cycle():
    with pytest.raises(ValueError):
        fws.Cycle(5, dim=3)

    F = fws.Cycle(5, dim=1)
    assert (
        F.dim == 1
        and F._graph.number_of_nodes() == 5
        and F._graph.number_of_edges() == 5
        and F.is_inf_rigid()
    )

    F = fws.Cycle(4, dim=2)
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 4
        and all([is_zero(L - sp.sympify("sqrt(2)")) for L in F.edge_lengths().values()])
        and len(F.inf_flexes()) == 1
    )

    F = fws.Cycle(6, dim=5)
    assert (
        F.dim == 5
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 6
        and all([L in [1, sp.sympify("sqrt(2)")] for L in F.edge_lengths().values()])
    )


def test_Square():
    F = fws.Square()
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 4
        and len(F.inf_flexes()) == 1
    )


def test_Diamond():
    F = fws.Diamond()
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 5
        and F.is_inf_rigid()
    )


def test_Cube():
    F = fws.Cube()
    assert (
        F.dim == 3
        and F._graph.number_of_nodes() == 8
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 6
    )


def test_Octahedron():
    F = fws.Octahedron()
    assert (
        F.dim == 3
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 0
    )

    F = fws.Octahedron(realization="Bricard_line")
    assert (
        F.dim == 3
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )

    F = fws.Octahedron(realization="Bricard_plane")
    assert (
        F.dim == 3
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )


def test_K33plusEdge():
    F = fws.K33plusEdge()
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 10
        and F.is_inf_rigid()
    )


def test_Complete():
    with pytest.raises(ValueError):
        fws.Complete(5, dim=3)

    F = fws.Complete(5, dim=1)
    assert (
        F.dim == 1
        and F._graph.number_of_nodes() == 5
        and F._graph.number_of_edges() == 10
        and F.is_inf_rigid()
    )

    F = fws.Complete(4, dim=2)
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 6
        and F.is_inf_rigid()
        and len(F.stresses()) == 1
    )

    F = fws.Complete(3, dim=3)
    assert (
        F.dim == 3
        and F._graph.number_of_nodes() == 3
        and F._graph.number_of_edges() == 3
        and F.is_inf_rigid()
        and len(F.stresses()) == 0
    )


def test_Path():
    with pytest.raises(ValueError):
        fws.Path(5, dim=3)

    F = fws.Path(5, dim=1)
    assert (
        F.dim == 1
        and F._graph.number_of_nodes() == 5
        and F._graph.number_of_edges() == 4
        and F.is_inf_rigid()
    )

    F = fws.Path(4, dim=2)
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 3
        and len(F.inf_flexes()) == 2
    )

    F = fws.Path(3, dim=3)
    assert (
        F.dim == 3
        and F._graph.number_of_nodes() == 3
        and F._graph.number_of_edges() == 2
        and len(F.inf_flexes()) == 1
    )


def test_ThreePrism():
    F = fws.ThreePrism()
    assert (
        F.dim == 2
        and F._graph.number_of_edges() == 9
        and F._graph.number_of_nodes() == 6
        and F.is_inf_rigid()
    )

    F = fws.ThreePrism(realization="flexible")
    assert (
        F.dim == 2
        and F._graph.number_of_edges() == 9
        and F._graph.number_of_nodes() == 6
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )

    F = fws.ThreePrism(realization="parallel")
    assert (
        F.dim == 2
        and F._graph.number_of_edges() == 9
        and F._graph.number_of_nodes() == 6
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )


def test_ThreePrismPlusEdge():
    F = fws.ThreePrismPlusEdge()
    assert (
        F.dim == 2
        and F._graph.number_of_edges() == 10
        and F._graph.number_of_nodes() == 6
        and F.is_inf_rigid()
    )


def test_CompleteBipartite():
    with pytest.raises(TypeError):
        fws.CompleteBipartite(1.5, 2.5)

    F = fws.CompleteBipartite(3, 4)
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 7
        and F._graph.number_of_edges() == 12
        and F.is_inf_rigid()
    )


def test_Frustum():
    with pytest.raises(ValueError):
        fws.Frustum(2)

    F = fws.Frustum(3)
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 9
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )

    F = fws.Frustum(4)
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 8
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 2
        and len(F.stresses()) == 1
    )


def test_CnSymmetricFourRegular():
    with pytest.raises(ValueError):
        fws.CnSymmetricFourRegular(6)
        fws.CnSymmetricFourRegular(9)

    F = fws.CnSymmetricFourRegular(8)
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 8
        and F._graph.number_of_edges() == 16
        and all(
            [
                is_zero(L - sp.sympify("sqrt((1 - sqrt(2)/2)**2 + 1/2)"))
                or is_zero(L - sp.sympify("sqrt((1 + sqrt(2)/2)**2 + 1/2)"))
                for L in F.edge_lengths().values()
            ]
        )
    ) and all([F._graph.degree[v] == 4 for v in F._graph.nodes])


def test_CnSymmetricWithFixedVertex():
    with pytest.raises(ValueError):
        fws.CnSymmetricWithFixedVertex(6)
        fws.CnSymmetricWithFixedVertex(9)

    F = fws.CnSymmetricWithFixedVertex(8)
    assert (
        F.dim == 2
        and F._graph.number_of_nodes() == 17
        and F._graph.number_of_edges() == 40
        and all(
            [
                any(
                    [
                        is_zero(L - z)
                        for z in sp.sympify(
                            [
                                "sqrt((1 - sqrt(2)/2)**2 + 1/2)",
                                "sqrt(106)/5",
                                "2*sqrt(53)/5",
                                "sqrt(1/2 + (sqrt(2)/2 + 1)**2)",
                                "sqrt(41)/5",
                            ]
                        )
                    ]
                )
                for L in F.edge_lengths().values()
            ]
        )
    ) and all([F._graph.degree[v] in [4, 5, 8] for v in F._graph.nodes])


def test_Icosahedron():
    F = fws.Icosahedron()
    G = graphs.Icosahedral()
    assert G.is_isomorphic(F._graph)
    edge_lengths = list(F.edge_lengths().values())
    for length in edge_lengths:
        assert sp.simplify(length - edge_lengths[0]) == 0


def test_Dodecahedron():
    F = fws.Dodecahedron()
    G = graphs.Dodecahedral()
    assert G.is_isomorphic(F._graph)
    edge_lengths = list(F.edge_lengths().values())
    for length in edge_lengths:
        assert sp.simplify(length - edge_lengths[0]) == 0


def test_SecondOrderRigid():
    F = fws.SecondOrderRigid()
    assert (
        F._dim == 3
        and F._graph.number_of_nodes() == 7
        and F._graph.number_of_edges() == 15
        and len(F.inf_flexes()) == 2
        and len(F.stresses()) == 2
    )


def test_Wheel():
    with pytest.raises(ValueError):
        fws.Wheel(1)
        fws.Wheel(2)

    for k in range(3, 10):
        F = fws.Wheel(k)
        G = graphs.Wheel(k)
        assert G.is_isomorphic(F.graph) and all(
            [
                any([is_zero(sum([p**2 for p in pos]) - val) for val in [0, 1]])
                for pos in F.realization(as_points=True).values()
            ]
        )
