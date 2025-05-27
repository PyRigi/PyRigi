import pytest

from pyrigi.data_type import Edge
from pyrigi.graph._flexibility.nac.core import (coloring_from_mask,
                                                mask_matches_templates)

ordered_comp_ids_common = [0, 1, 2]
class_to_edges_common: list[list[Edge]] = [
    [("a", "b"), ("b", "c")],
    [("d", "e")],
    [("f", "g"), ("g", "h")],
]


def test_coloring_from_mask_basic_and_symmetry():
    mask = 0b101
    expected_red = [("a", "b"), ("b", "c"), ("f", "g"), ("g", "h")]
    expected_blue = [("d", "e")]

    result_red, result_blue = coloring_from_mask(
        ordered_comp_ids_common, class_to_edges_common, mask
    )
    assert sorted(result_red) == sorted(expected_red)
    assert sorted(result_blue) == sorted(expected_blue)

    num_components = len(ordered_comp_ids_common)
    inverted_mask = (~mask) & ((1 << num_components) - 1)

    result_inverted_red, result_inverted_blue = coloring_from_mask(
        ordered_comp_ids_common, class_to_edges_common, inverted_mask
    )
    assert sorted(result_inverted_red) == sorted(expected_blue)
    assert sorted(result_inverted_blue) == sorted(expected_red)


def test_coloring_from_mask_allow_mask_and_symmetry():
    mask = 0b101
    allow_mask = 0b011
    expected_red = [("a", "b"), ("b", "c")]
    expected_blue = [("d", "e")]

    result_red, result_blue = coloring_from_mask(
        ordered_comp_ids_common, class_to_edges_common, mask, allow_mask
    )
    assert sorted(result_red) == sorted(expected_red)
    assert sorted(result_blue) == sorted(expected_blue)

    inverted_mask_for_allowed = (~mask) & allow_mask

    result_inverted_red, result_inverted_blue = coloring_from_mask(
        ordered_comp_ids_common,
        class_to_edges_common,
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


################################################################################
@pytest.mark.parametrize(
    "templates, mask, subgraph_mask, expected",
    [
        ([], 0b0010, 0b0010, False),
        ([(0b0011, 0b0001)], 0b0100, 0b0100, False),
        ([(0b0001, 0b0010)], 0b0001, 0b0001, False),
        ([(0b0001, 0b0001)], 0b0001, 0b0001, True),
        ([(0b0011, 0b0010)], 0b0000, 0b0010, True),
        ([(0b0001, 0b0001), (0b1000, 0b1000)], 0b0001, 0b0000, True),
        ([(0b0010, 0b0100), (0b1000, 0b1000)], 0b1000, 0b0000, True),
        ([(0b0001, 0b0001)], 0, 0, False),
        ([(0b0001, 0b0001)], 0b1111, 0b1111, True),
        ([(0b0100, 0b0100)], 0b0001, 0b0011, False),
        ([(0b0010, 0b0010)], 0b0010, 0b0110, True),
        ([(0b0110, 0b0100)], 0b0110, 0b0010, True),
        ([(0b0001, 0b0010), (0b0100, 0b0100)], 0b0001, 0b0000, False),
        ([(0b1010, 0b1000)], 0b1010, 0b0010, True),
        ([(0b00000001, 0b00000001)], 0b00000001, 0b11111111, True),
        ([(0b00000100, 0b00000100)], 0b00000010, 0b11111111, True),
        ([(0b00010000, 0b00010000)], 0b00010000, 0b00001111, True),
        ([(0b00000011, 0b00000001)], 0b00000011, 0b11111111, False),
        ([(0b00000001, 0b00000010)], 0b00000001, 0b11111111, False),
    ],
)
def test_mask_matches_templates(templates, mask, subgraph_mask, expected):
    result = mask_matches_templates(templates, mask, subgraph_mask)
    assert result == expected
