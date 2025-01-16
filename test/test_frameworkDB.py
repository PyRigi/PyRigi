import pyrigi.frameworkDB as fws
import numpy as np
from math import isclose

import pytest


def test_Cycle():
    with pytest.raises(ValueError):
        fws.Cycle(5, dim=3)

    F = fws.Cycle(5, dim=1)
    assert (
        F._dim == 1
        and F._graph.number_of_nodes() == 5
        and F._graph.number_of_edges() == 5
        and F.is_inf_rigid()
    )

    F = fws.Cycle(4, dim=2)
    assert (
        F._dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 4
        and all(
            [
                np.linalg.norm(p) == 1
                for p in F.realization(as_points=True, numerical=True).values()
            ]
        )
        and len(F.inf_flexes()) == 1
    )

    F = fws.Cycle(6, dim=5)
    assert (
        F._dim == 5
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 6
        and all(
            [
                sum(p) == 0 or sum(p) == 1
                for p in F.realization(as_points=True, numerical=True).values()
            ]
        )
    )


def test_Square():
    F = fws.Square()
    assert (
        F._dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 4
        and len(F.inf_flexes()) == 1
    )


def test_Diamond():
    F = fws.Diamond()
    assert (
        F._dim == 2
        and F.is_congruent_realization(fws.Square().realization(as_points=True))
        and F.is_equivalent_realization(fws.Square().realization(as_points=True))
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 5
        and F.is_inf_rigid()
    )


def test_Cube():
    F = fws.Cube()
    assert (
        F._dim == 3
        and F._graph.number_of_nodes() == 8
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 6
    )


def test_Octahedron():
    F = fws.Octahedron()
    assert (
        F._dim == 3
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 0
    )

    F = fws.Octahedron(realization="Bricard_line")
    assert (
        F._dim == 3
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )

    F = fws.Octahedron(realization="Bricard_plane")
    assert (
        F._dim == 3
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 12
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )


def test_K33plusEdge():
    F = fws.K33plusEdge()
    assert (
        F._dim == 2
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 10
        and F.is_inf_rigid()
    )


def test_Complete():
    with pytest.raises(ValueError):
        fws.Complete(5, dim=3)

    F = fws.Complete(5, dim=1)
    assert (
        F._dim == 1
        and F._graph.number_of_nodes() == 5
        and F._graph.number_of_edges() == 10
        and F.is_inf_rigid()
    )

    F = fws.Complete(4, dim=2)
    assert (
        F._dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 6
        and F.is_inf_rigid()
        and len(F.stresses()) == 1
    )

    F = fws.Complete(3, dim=3)
    assert (
        F._dim == 3
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
        F._dim == 1
        and F._graph.number_of_nodes() == 5
        and F._graph.number_of_edges() == 4
        and F.is_inf_rigid()
    )

    F = fws.Path(4, dim=2)
    assert (
        F._dim == 2
        and F._graph.number_of_nodes() == 4
        and F._graph.number_of_edges() == 3
        and len(F.inf_flexes()) == 2
    )

    F = fws.Path(3, dim=3)
    assert (
        F._dim == 3
        and F._graph.number_of_nodes() == 3
        and F._graph.number_of_edges() == 2
        and len(F.inf_flexes()) == 1
    )


def test_ThreePrism():
    F = fws.ThreePrism()
    assert (
        F._dim == 2
        and F._graph.number_of_edges() == 9
        and F._graph.number_of_nodes() == 6
        and F.is_inf_rigid()
    )

    F = fws.ThreePrism(realization="flexible")
    assert (
        F._dim == 2
        and F._graph.number_of_edges() == 9
        and F._graph.number_of_nodes() == 6
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )

    F = fws.ThreePrism(realization="parallel")
    assert (
        F._dim == 2
        and F._graph.number_of_edges() == 9
        and F._graph.number_of_nodes() == 6
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )


def test_ThreePrismPlusEdge():
    F = fws.ThreePrismPlusEdge()
    assert (
        F._dim == 2
        and F._graph.number_of_edges() == 10
        and F._graph.number_of_nodes() == 6
        and F.is_inf_rigid()
    )


def test_CompleteBipartite():
    with pytest.raises(TypeError):
        fws.CompleteBipartite(1.5, 2.5)

    F = fws.CompleteBipartite(3, 4)
    print(
        [
            [np.linalg.norm(p), np.linalg.norm([p[0] - 1, p[1] - 3])]
            for p in F.realization(as_points=True, numerical=True).values()
        ]
    )
    assert (
        F._dim == 2
        and F._graph.number_of_nodes() == 7
        and F._graph.number_of_edges() == 12
        and F.is_inf_rigid()
        and all(
            [
                isclose(np.linalg.norm(p), 1, rel_tol=1e-10)
                or isclose(np.linalg.norm([p[0] - 1, p[1] - 3]), 2, rel_tol=1e-10)
                for p in F.realization(as_points=True, numerical=True).values()
            ]
        )
    )


def test_Frustum():
    with pytest.raises(ValueError):
        fws.Frustum(2)

    F = fws.Frustum(3)
    assert (
        F._dim == 2
        and F._graph.number_of_nodes() == 6
        and F._graph.number_of_edges() == 9
        and len(F.inf_flexes()) == 1
        and len(F.stresses()) == 1
    )

    F = fws.Frustum(4)
    assert (
        F._dim == 2
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
        F._dim == 2
        and F._graph.number_of_nodes() == 8
        and F._graph.number_of_edges() == 16
        and all(
            [
                np.linalg.norm(p) == 1
                for p in F.realization(as_points=True, numerical=True).values()
            ]
        )
    ) and all([F._graph.degree[v] == 4 for v in F._graph.nodes])


def test_CnSymmetricFourRegularWithFixedVertex():
    with pytest.raises(ValueError):
        fws.CnSymmetricFourRegularWithFixedVertex(6)
        fws.CnSymmetricFourRegularWithFixedVertex(9)

    F = fws.CnSymmetricFourRegularWithFixedVertex(8)
    assert (
        F._dim == 2
        and F._graph.number_of_nodes() == 17
        and F._graph.number_of_edges() == 40
        and all(
            [
                np.linalg.norm(p) in [0, 1]
                or isclose(np.linalg.norm(p), 2.0591260281974, rel_tol=1e-8)
                for p in F.realization(as_points=True, numerical=True).values()
            ]
        )
    ) and all([F._graph.degree[v] in [4, 5, 8] for v in F._graph.nodes])
