from pyrigi.matroid import LinearMatroid
from sympy import Matrix


def test_linear_matroid_ground_set():
    A = Matrix([[1, 2], [3, 4], [5, 6]])
    M = LinearMatroid(A)
    assert M.ground_set() == [0, 1, 2]


def test_linear_matroid_rank():
    A = Matrix([[1, 2], [3, 4], [5, 6]])
    M = LinearMatroid(A)
    assert M.rank() == 2
    assert M.rank([0, 1]) == 2
    assert M.rank([0]) == 1


def test_linear_matroid_independence():
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 4, 5]])
    M = LinearMatroid(A)
    assert M.is_independent([0, 2])
    assert M.is_dependent([0, 1])
    assert M.is_circuit([0, 1])
    assert M.is_basis([0, 2])
