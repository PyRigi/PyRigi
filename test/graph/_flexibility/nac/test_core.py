from pyrigi.graph._flexibility.nac.core import coloring_from_mask
from pyrigi.data_type import Edge


ordered_comp_ids_common = [0, 1, 2]
component_to_edges_common: list[list[Edge]] = [
    [("a", "b"), ("b", "c")],
    [("d", "e")],
    [("f", "g"), ("g", "h")],
]


def test_coloring_from_mask_basic_and_symmetry():
    mask = 0b101
    expected_red = [("a", "b"), ("b", "c"), ("f", "g"), ("g", "h")]
    expected_blue = [("d", "e")]

    result_red, result_blue = coloring_from_mask(
        ordered_comp_ids_common, component_to_edges_common, mask
    )
    assert sorted(result_red) == sorted(expected_red)
    assert sorted(result_blue) == sorted(expected_blue)

    num_components = len(ordered_comp_ids_common)
    inverted_mask = (~mask) & ((1 << num_components) - 1)

    result_inverted_red, result_inverted_blue = coloring_from_mask(
        ordered_comp_ids_common, component_to_edges_common, inverted_mask
    )
    assert sorted(result_inverted_red) == sorted(expected_blue)
    assert sorted(result_inverted_blue) == sorted(expected_red)


def test_coloring_from_mask_allow_mask_and_symmetry():
    mask = 0b101
    allow_mask = 0b011
    expected_red = [("a", "b"), ("b", "c")]
    expected_blue = [("d", "e")]

    result_red, result_blue = coloring_from_mask(
        ordered_comp_ids_common, component_to_edges_common, mask, allow_mask
    )
    assert sorted(result_red) == sorted(expected_red)
    assert sorted(result_blue) == sorted(expected_blue)

    inverted_mask_for_allowed = (~mask) & allow_mask

    result_inverted_red, result_inverted_blue = coloring_from_mask(
        ordered_comp_ids_common,
        component_to_edges_common,
        inverted_mask_for_allowed,
        allow_mask,
    )
    assert sorted(result_inverted_red) == sorted(expected_blue)
    assert sorted(result_inverted_blue) == sorted(expected_red)


def test_coloring_from_mask_empty_inputs():
    ordered_comp_ids_empty = []
    component_to_edges_empty = []
    mask_empty = 0b000
    expected_red_empty = []
    expected_blue_empty = []
    result_red, result_blue = coloring_from_mask(
        ordered_comp_ids_empty, component_to_edges_empty, mask_empty
    )
    assert sorted(result_red) == sorted(expected_red_empty)
    assert sorted(result_blue) == sorted(expected_blue_empty)


def test_coloring_from_mask_component_with_no_edges():
    ordered_comp_ids = [1, 0]
    component_to_edges = [[("x", "y")], []]
    mask = 0b10
    expected_red = [("x", "y")]
    expected_blue = []
    result_red, result_blue = coloring_from_mask(
        ordered_comp_ids, component_to_edges, mask
    )
    assert sorted(result_red) == sorted(expected_red)
    assert sorted(result_blue) == sorted(expected_blue)
