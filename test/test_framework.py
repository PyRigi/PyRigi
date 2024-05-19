from pyrigi.graph import Graph
from pyrigi.framework import Framework

from sympy import Matrix


def test_dimension():
    F = Framework(Graph([[0, 1]]), {0: [1, 2], 1: [0, 5]})
    assert F.dim() == F.dimension()
    assert F.dim() == 2
    F_ = Framework.Empty(dim=3)
    assert F_.dim() == 3


def test_vertex_addition():
    F = Framework.Empty()
    F.add_vertices([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_ = Framework.Empty()
    F_.add_vertices([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_.set_realization(F.realization())
    assert (
        F.get_realization_list() == F_.get_realization_list()
        and F.graph().vertex_list() == F_.graph().vertex_list()
        and F.dim() == F_.dim()
    )
    assert F.graph().vertex_list() == [0, 1, 2] and len(F.graph().edges()) == 0
    F.change_vertex_coordinates_list([0, 2], [[3.0, 0.0], [0.0, 3.0]])
    F_.change_vertex_coordinates(1, [2.0, 2.0])
    array = F_.get_realization_list()
    array[0] = (3, 0)
    assert F[0] != F_[0] and F[1] != F_[1] and F[2] != F_[2]


def test_inf_rigidity():
    F = Framework(Graph([[0, 1], [1, 2], [0, 2]]), {0: [1, 2], 1: [0, 5], 2: [3, 3]})
    assert F.is_inf_rigid() and F.is_min_inf_rigid()
    F_ = Framework(
        Graph([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]),
        {0: [0, 0], 1: [2, 0], 2: [2, 2], 3: [-1, 1]},
    )
    assert F_.is_inf_rigid() and not F_.is_min_inf_rigid()
    F_.delete_edge([0, 2])
    assert F_.is_inf_rigid() and F_.is_min_inf_rigid()


def test_inf_flexes():
    F = Framework(Graph([[0, 1]]), {0: [0, 0], 1: [1, 0]})
    Q1 = Matrix.hstack(*(F.inf_flexes(include_trivial=True)))
    Q2 = Matrix.hstack(*(F.trivial_inf_flexes()))
    assert Q1.rank() == Q2.rank() and Q1.rank() == Matrix.hstack(Q1, Q2).rank()
    F = Framework(
        Graph([[0, 1], [1, 2], [2, 3], [0, 3]]),
        {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]},
    )
    assert len(F.inf_flexes(include_trivial=False)) == 1
