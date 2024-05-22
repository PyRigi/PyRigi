from pyrigi.framework import Framework
import pyrigi.frameworkDB as fws

from sympy import Matrix


def test_dimension():
    assert fws.Complete(2, 2).dim() == fws.Complete(2, 2).dimension()
    assert fws.Complete(2, 2).dim() == 2
    assert Framework.Empty(dim=3).dim() == 3


def test_vertex_addition():
    F = Framework.Empty()
    F.add_vertices([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_ = Framework.Empty()
    F_.add_vertices([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_.set_realization(F.realization())
    assert (
        F.realization() == F_.realization()
        and F.graph().vertex_list() == F_.graph().vertex_list()
        and F.dim() == F_.dim()
    )
    assert F.graph().vertex_list() == [0, 1, 2] and len(F.graph().edges()) == 0
    F.set_vertex_positions_from_lists([0, 2], [[3.0, 0.0], [0.0, 3.0]])
    F_.set_vertex_pos(1, [2.0, 2.0])
    array = F_.realization()
    array[0] = (3, 0)
    assert F[0] != F_[0] and F[1] != F_[1] and F[2] != F_[2]


def test_inf_rigidity():
    assert fws.Complete(3, 2).is_inf_rigid() and fws.Complete(3, 2).is_min_inf_rigid()
    assert (
        fws.Complete(4, 2).is_inf_rigid() and not fws.Complete(4, 2).is_min_inf_rigid()
    )
    assert fws.Diamond().is_inf_rigid() and fws.Diamond().is_min_inf_rigid()


def test_inf_flexes():
    Q1 = Matrix.hstack(*(fws.Complete(2, 2).inf_flexes(include_trivial=True)))
    Q2 = Matrix.hstack(*(fws.Complete(2, 2).trivial_inf_flexes()))
    assert Q1.rank() == Q2.rank() and Q1.rank() == Matrix.hstack(Q1, Q2).rank()
    assert len(fws.Square().inf_flexes(include_trivial=False)) == 1
