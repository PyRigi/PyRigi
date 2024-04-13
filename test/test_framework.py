from pyrigi.graph import Graph
from pyrigi.framework import Framework

def test_dimension():
    F = Framework(Graph([[0,1]]),{0:[1,2], 1:[0,5]})
    assert F.dim() == F.dimension()
    assert F.dim() == 2
    F = Framework(dim=3)
    assert F.dim() == 3

def test_vertex_addition():
    F = Framework()
    F.add_vertices([[1., 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_ = Framework()
    F_.add_vertices([[1., 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_.set_realization(F.get_realization())
    assert (
        F.get_realization_list() == F_.get_realization_list()
        and F.graph().vertices() == F_.graph().vertices()
        and F.dim() == F_.dim()
    )
    assert (
        list(F.graph().vertices()) == [0, 1, 2]
        and len(F.graph().edges()) == 0
    )
    F.change_vertex_coordinates_list([0, 2], [[3.,0.], [0.,3.]])
    F_.change_vertex_coordinates(1, [2.,2.])
    array = F_.get_realization_list()
    array[0] = (3,0)
    assert(
        F.get_realization()[0] != F_.get_realization()[0]
        and F.get_realization()[1] != F_.get_realization()[1]
        and F.get_realization()[2] != F_.get_realization()[2]
    )
    F_.add_edge([0, 1])
    plt = F_.draw_framework()