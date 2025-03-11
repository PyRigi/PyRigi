import pytest

import pyrigi.graphDB as graphs


def test_Cycle():
    for n in range(3, 10):
        G = graphs.Cycle(n)
        assert (
            G.number_of_nodes() == n
            and G.number_of_edges() == n
            and all([G.degree[v] == 2 for v in G.nodes])
        )


def test_Complete():
    for n in range(1, 10):
        G = graphs.Complete(n)
        assert (
            G.number_of_nodes() == n
            and G.number_of_edges() == n * (n - 1) / 2
            and all([G.degree[v] == n - 1 for v in G.nodes])
        )


def test_Path():
    for n in range(2, 10):
        G = graphs.Path(n)
        assert (
            G.number_of_nodes() == n
            and G.number_of_edges() == n - 1
            and all([G.degree[v] in [1, 2] for v in G.nodes])
        )


def test_CompleteBipartite():
    for n in range(1, 10):
        for m in range(1, 10):
            G = graphs.CompleteBipartite(n, m)
            assert (
                G.number_of_nodes() == n + m
                and G.number_of_edges() == m * n
                and all([G.degree[v] in [n, m] for v in G.nodes])
            )


def test_K33plusEdge():
    G = graphs.K33plusEdge()
    assert (
        G.number_of_nodes() == 6
        and G.number_of_edges() == 10
        and all([G.degree[v] in [3, 4] for v in G.nodes])
    )


def test_Diamond():
    G = graphs.Diamond()
    assert (
        G.number_of_nodes() == 4
        and G.number_of_edges() == 5
        and all([G.degree[v] in [2, 3] for v in G.nodes])
    )


def test_ThreePrism():
    G = graphs.ThreePrism()
    assert (
        G.number_of_edges() == 9
        and G.number_of_nodes() == 6
        and all([G.degree[v] == 3 for v in G.nodes])
    )


def test_ThreePrismPlusEdge():
    G = graphs.ThreePrismPlusEdge()
    assert (
        G.number_of_edges() == 10
        and G.number_of_nodes() == 6
        and all([G.degree[v] in [3, 4] for v in G.nodes])
    )


def test_CubeWithDiagonal():
    G = graphs.CubeWithDiagonal()
    assert (
        G.number_of_nodes() == 8
        and G.number_of_edges() == 13
        and all([G.degree[v] in [3, 4] for v in G.nodes])
    )


def test_DoubleBanana():
    with pytest.raises(ValueError):
        graphs.DoubleBanana(dim=2)
        graphs.DoubleBanana(t=3)
        graphs.DoubleBanana(dim=4, t=4)

    for dim in range(3, 8):
        for t in range(2, dim):
            G = graphs.DoubleBanana(dim=dim, t=t)
            assert (
                G.number_of_nodes() == (2 + dim) * 2 - t
                and G.number_of_edges() == dim * (dim + 3) + 1 - t * (t - 1) / 2
                and all(
                    [
                        G.degree[v] in [dim + 1, 2 * dim + 2 - t, 2 * dim + 2 - t + 1]
                        for v in G.nodes
                    ]
                )
            )


def test_CnSymmetricFourRegular():
    with pytest.raises(ValueError):
        graphs.CnSymmetricFourRegular(6)
        graphs.CnSymmetricFourRegular(9)

    for i in range(4, 10):
        G = graphs.CnSymmetricFourRegular(2 * i)
        assert (
            G.number_of_nodes() == 2 * i
            and G.number_of_edges() == 4 * i
            and all([G.degree[v] == 4 for v in G.nodes])
        )


def test_CnSymmetricWithFixedVertex():
    with pytest.raises(ValueError):
        graphs.CnSymmetricWithFixedVertex(6)
        graphs.CnSymmetricWithFixedVertex(9)

    for i in range(4, 10):
        G = graphs.CnSymmetricWithFixedVertex(2 * i)
        assert (
            G.number_of_nodes() == 4 * i + 1
            and G.number_of_edges() == 10 * i
            and all([G.degree[v] in [4, 5, 2 * i] for v in G.nodes])
        )


def test_Icosahedral():
    G = graphs.Icosahedral()
    assert G.number_of_nodes() == 12 and G.number_of_edges() == 30


def test_Dodecahedron():
    G = graphs.Dodecahedral()
    assert G.number_of_nodes() == 20 and G.number_of_edges() == 30


def test_Wheel():
    with pytest.raises(ValueError):
        graphs.Wheel(1)
        graphs.Wheel(2)

    for k in range(3, 10):
        G = graphs.Wheel(k)
        assert (
            G.number_of_nodes() == k + 1
            and G.number_of_edges() == 2 * k
            and all([G.degree[v] in [3, k] for v in G.nodes])
        )
