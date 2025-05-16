import pytest

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi import ApproximateMotion, Graph, Motion, ParametricMotion


def test__str__():
    assert (
        str(Motion(graphs.Complete(3), dim=2))
        == "Motion of a Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]"
    )
    assert (
        str(
            ParametricMotion(
                graphs.Path(3),
                {0: [-1, 0], 1: [0, 0], 2: ["sin(t)", "cos(t)"]},
                [-1, 1],
            )
        )
        == "ParametricMotion of a Graph with vertices [0, 1, 2] "
        """and edges [[0, 1], [1, 2]] with motion defined for every vertex:
0: Matrix([[-1], [0]])
1: Matrix([[0], [0]])
2: Matrix([[sin(t)], [cos(t)]])"""
    )  # noqa: E501
    assert (
        str(ApproximateMotion(fws.Path(3), 2))
        == "ApproximateMotion of a Graph with vertices [0, 1, 2]"
        """ and edges [[0, 1], [1, 2]] with starting configuration
{0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]},
2 retraction steps and initial step size 0.05."""
    )


@pytest.mark.parametrize(
    "motion, motion_repr",
    [
        [
            Motion(graphs.Complete(3), dim=2),
            "Motion(Graph.from_vertices_and_edges([0, 1, 2],"
            " [(0, 1), (0, 2), (1, 2)]), 2)",
        ],
        [
            ParametricMotion(
                graphs.Path(3),
                {0: [-1, 0], 1: [0, 0], 2: ["sin(t)", "cos(t)"]},
                [-1, 1],
            ),
            "ParametricMotion(Graph.from_vertices_and_edges([0, 1, 2], [(0, 1), (1, 2)]),"
            " {0: ['-1', '0'], 1: ['0', '0'], 2: ['sin(t)', 'cos(t)']}, [-1, 1])",
        ],
        [
            ApproximateMotion(fws.Path(3), 2),
            "ApproximateMotion.from_graph(Graph.from_vertices_and_edges([0, 1, 2], "
            "[(0, 1), (1, 2)]), {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]},"
            " 2, step_size=0.05, chosen_flex=0, tolerance=1e-05, fixed_pair=None,"
            " fixed_direction=None, pinned_vertex=0)",
        ],
    ],
)
def test__repr__(motion, motion_repr):
    eval(repr(motion))
    assert repr(motion) == motion_repr
