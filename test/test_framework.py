from pyrigi.graph import Graph
from pyrigi.framework import Framework

def test_dimension():
    F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
    assert F.dim() == F.dimension()
    assert F.dim() == 2
    F_ = Framework(dim=3)
    assert F_.dim() == 3

def test_vertex_addition():
    F = Framework()
    F.add_vertices([[1., 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_ = Framework()
    F_.add_vertices([[1., 2.0], [1.0, 1.0], [0.0, 0.0]])
    F_.set_realization(F.get_realization())
    assert (
        F.get_realization_list() == F_.get_realization_list()
        and F.graph().vertex_list() == F_.graph().vertex_list()
        and F.dim() == F_.dim()
    )
    assert (
        F.graph().vertex_list() == [0, 1, 2]
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

def test_inf_rigidity():
    F = Framework(Graph([[0,1], [1,2], [0,2]]), {0:[1,2], 1:[0,5], 2:[3,3]})
    assert (F.is_infinitesimally_rigid() and F.is_minimally_infinitesimally_rigid())
    F_ = Framework(Graph([[0,1], [1,2], [2,3], [3,0], [0,2], [1,3]]), {0:[0,0], 1:[2,0], 2:[2,2], 3:[-1,1]})
    assert (F_.is_infinitesimally_rigid() and not F_.is_minimally_infinitesimally_rigid())
    F_.delete_edge([0,2])
    assert (F_.is_infinitesimally_rigid() and F_.is_minimally_infinitesimally_rigid())

def test_inf_motions():
    F = Framework(Graph([[0,1]]), {0:[0,0], 1:[1,0]})
    assert F.infinitesimal_flexes(include_trivial=True) == F.trivial_infinitesimal_flexes()
    F_ = Framework(Graph([[0,1], [1,2], [2,3], [0,3]]), {0:[0,0], 1:[1,0], 2:[1,1], 3:[0,1]})
    assert (len(F_.infinitesimal_flexes(include_trivial=False))==1 and F_.infinitesimal_flexes(include_trivial=False)[0].rank()==1)

def test_pinning():
    F = Framework(Graph([[0,1], [0,2]]), {0:[0,0], 1:[1,0], 2:[1,1]})
    assert(len(F.infinitesimal_flexes(pinned_vertices={0:[0,1],1:[1]}, include_trivial=False))==1 and
            len(F.infinitesimal_flexes(pinned_vertices={0:[0,1],1:[1]}, include_trivial=True))==1 and
            F.infinitesimal_flexes(pinned_vertices={0:[0,1],1:[1]}, include_trivial=True)[0][0:4]==[0 for _ in range(4)])
    F.add_edge([1,2])
    assert(len(F.infinitesimal_flexes(pinned_vertices={0:[0,1],1:[1]}, include_trivial=False))==0 and
           len(F.infinitesimal_flexes(pinned_vertices={0:[0,1],1:[1]}, include_trivial=True))==0)
    assert(len(F.infinitesimal_flexes(pinned_vertices={0:[0],2:[0]}, include_trivial=False))==0 and
           len(F.infinitesimal_flexes(pinned_vertices={0:[0],2:[0]}, include_trivial=True))==1)
