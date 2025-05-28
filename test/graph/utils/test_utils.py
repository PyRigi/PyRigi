import pytest

from pyrigi import Graph
from pyrigi.graph._utils.utils import is_isomorphic_graph_list


@pytest.mark.parametrize(
    "list1, list2",
    [
        [[239, 254, 254], [254, 254, 239]],
        [[239, 254], [254, 239]],
        [[31], [31]],
        [[254], [947]],
    ],
)
def test_is_isomorphic_graph_list(list1, list2):
    assert is_isomorphic_graph_list(
        [Graph.from_int(g) for g in list1], [Graph.from_int(g) for g in list2]
    )


@pytest.mark.parametrize(
    "list1, list2",
    [
        [[239, 254, 254], [254, 31, 239]],
        [[239, 254, 254], [254, 239, 239]],
        [[239, 254, 254], [254, 239]],
        [[239, 254], [31, 239]],
    ],
)
def test_is_not_isomorphic_graph_list(list1, list2):
    assert not is_isomorphic_graph_list(
        [Graph.from_int(g) for g in list1], [Graph.from_int(g) for g in list2]
    )
