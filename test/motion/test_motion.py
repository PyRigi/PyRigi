from copy import deepcopy
from math import isclose

import numpy as np
import pytest
import sympy as sp

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi import ApproximateMotion, Framework, Graph, Motion, ParametricMotion
from pyrigi.misc.misc import is_zero_vector


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


def test__input_check_edge_lengths():
    motion = ParametricMotion(
        graphs.Cycle(4),
        {
            0: ("0", "0"),
            1: ("1", "0"),
            2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
            3: (
                "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
                "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
            ),
        },
        [-sp.oo, sp.oo],
    )
    motion._input_check_edge_lengths()

    t = sp.Symbol("t")
    motion = ParametricMotion(
        graphs.Cycle(4),
        {
            0: (0, 0),
            1: (1, 0),
            2: (4 * (t**2 - 2) / (t**2 + 4), 12 * t / (t**2 + 4)),
            3: (
                (t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4),
                6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4),
            ),
        },
        [-sp.oo, sp.oo],
    )
    motion._input_check_edge_lengths()

    motion = {
        0: ("t", "0"),
        1: ("1", "0"),
        2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
        3: (
            "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
            "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
        ),
    }
    with pytest.raises(ValueError):
        ParametricMotion(graphs.Cycle(4), motion, [-sp.oo, sp.oo])

    a, b, d = 1, 3, 2
    t = sp.Symbol("t")
    sqrt_x = sp.sqrt(b**2 - a**2 * sp.sin(t) ** 2)
    sqrt_y = sp.sqrt(d**2 - a**2 * sp.cos(t) ** 2)
    p = {
        0: [a * sp.cos(t) + sqrt_x, a * sp.sin(t) + sqrt_y],
        1: [-a * sp.cos(t) - sqrt_x, a * sp.sin(t) + sqrt_y],
        2: [-a * sp.cos(t) - sqrt_x, -a * sp.sin(t) - sqrt_y],
        3: [a * sp.cos(t) + sqrt_x, -a * sp.sin(t) - sqrt_y],
        4: [-a * sp.cos(t) + sqrt_x, -a * sp.sin(t) + sqrt_y],
        5: [a * sp.cos(t) - sqrt_x, -a * sp.sin(t) + sqrt_y],
        6: [a * sp.cos(t) - sqrt_x, a * sp.sin(t) - sqrt_y],
        7: [-a * sp.cos(t) + sqrt_x, a * sp.sin(t) - sqrt_y],
    }

    motion = ParametricMotion(graphs.CompleteBipartite(4, 4), p, [-sp.pi, sp.pi])
    motion._input_check_edge_lengths()


def test_realization():
    motion = ParametricMotion(
        graphs.Cycle(4),
        {
            0: ("0", "0"),
            1: ("1", "0"),
            2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
            3: (
                "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
                "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
            ),
        },
        [-sp.oo, sp.oo],
    )
    R = motion.realization(0, numerical=False)
    tmp = R[0]
    assert tmp[0] == 0
    assert tmp[1] == 0

    tmp = R[1]
    assert tmp[0] == 1
    assert tmp[1] == 0

    tmp = R[2]
    assert tmp[0] == -2
    assert tmp[1] == 0

    tmp = R[3]
    assert tmp[0] == 1
    assert tmp[1] == 0

    R = motion.realization("2/3", numerical=False)
    tmp = R[2]
    assert tmp[0] == sp.sympify("-7/5")
    assert tmp[1] == sp.sympify("9/5")

    tmp = R[3]
    assert tmp[0] == sp.sympify("-16/65")
    assert tmp[1] == sp.sympify("-63/65")

    R = motion.realization(2 / 3, numerical=True)
    tmp = R[2]
    assert abs(tmp[0] - (-7 / 5)) < 1e-9
    assert abs(tmp[1] - 9 / 5) < 1e-9

    tmp = R[3]
    assert abs(tmp[0] - (-16 / 65)) < 1e-9
    assert abs(tmp[1] - (-63 / 65)) < 1e-9


def test_ParametricMotion_init():
    motion = {
        0: [
            "t",
        ],
        1: [
            "t",
        ],
    }
    motion = ParametricMotion(graphs.Path(2), motion, [-10, 10])
    motion.animate(animation_format="svg")
    motion.animate(animation_format="matplotlib")

    motion = {
        0: ("k", "0"),
        1: ("1", "0"),
        2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
        3: (
            "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
            "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
        ),
    }
    with pytest.raises(ValueError):
        ParametricMotion(graphs.Cycle(4), motion, [-sp.oo, sp.oo])

    motion = {
        7: ("0", "0"),
        1: ("1", "0"),
        2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
        3: (
            "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
            "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
        ),
    }
    with pytest.raises(KeyError):
        ParametricMotion(graphs.Cycle(4), motion, [-sp.oo, sp.oo])

    t = 0
    motion = {
        0: (0, 0),
        1: (1, 0),
        2: (4 * (t**2 - 2) / (t**2 + 4), 12 * t / (t**2 + 4)),
        3: (
            (t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4),
            6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4),
        ),
    }

    with pytest.raises(ValueError):
        ParametricMotion(graphs.Cycle(4), motion, [-sp.oo, sp.oo])


@pytest.mark.parametrize(
    "framework",
    [
        Framework(Graph([(0, 1), (2, 3)]), {0: [0], 1: [1], 2: [2], 3: [3]}),
        fws.Square(),
        fws.Cycle(5),
        fws.Cycle(6),
        fws.ThreePrism("flexible"),
        fws.CompleteBipartite(2, 4),
        fws.CompleteBipartite(2, 5),
    ],
)
def test_animate(framework):
    motion = ApproximateMotion(framework, 5, 0.075)
    motion.animate(animation_format="svg")
    motion.animate(animation_format="matplotlib")


def test_animate3D():
    F = fws.Cube()
    motion = ApproximateMotion(F, 5, 0.075)
    for sample in motion.motion_samples[1:]:
        assert F.is_equivalent_realization(
            sample, numerical=True, tolerance=1e-3
        ) and not F.is_congruent_realization(sample, numerical=True)
    motion.animate()


@pytest.mark.parametrize(
    "framework",
    [
        Framework(Graph([(0, 1), (2, 3)]), {0: [0], 1: [1], 2: [2], 3: [3]}),
        fws.Square(),
        fws.Cycle(5),
        fws.Cycle(6),
        fws.ThreePrism("flexible"),
        fws.CompleteBipartite(2, 4),
        Framework(
            graphs.Complete(4) + Graph([(2, 4)]),
            fws.Complete(4).realization(as_points=True, numerical=True) | {4: [2, 2]},
        ),
        fws.CompleteBipartite(2, 5),
    ],
)
def test_ApproximateMotion_from_framework(framework):
    motion1 = ApproximateMotion(framework, 5, 0.075)
    for sample in motion1.motion_samples[1:]:
        assert framework.is_equivalent_realization(
            sample, numerical=True, tolerance=1e-3
        ) and not framework.is_congruent_realization(sample, numerical=True)

    try:
        motion2 = ApproximateMotion(
            framework, 5, 0.075, fixed_pair=(0, 1), fixed_direction=[1, 0]
        )
        for sample in motion2.motion_samples[1:]:
            assert framework.is_equivalent_realization(
                sample, numerical=True, tolerance=1e-3
            ) and not framework.is_congruent_realization(sample, numerical=True)
    except ValueError:
        assert framework._dim == 1


@pytest.mark.parametrize(
    "framework",
    [
        Framework(Graph([(0, 1), (2, 3)]), {0: [0], 1: [1], 2: [2], 3: [3]}),
        fws.Square(),
        fws.Cycle(5),
        fws.Cycle(6),
        fws.ThreePrism("flexible"),
        fws.CompleteBipartite(2, 4),
        Framework(
            graphs.Complete(4) + Graph([(2, 4)]),
            fws.Complete(4).realization(as_points=True, numerical=True) | {4: [2, 2]},
        ),
        fws.CompleteBipartite(2, 5),
    ],
)
def test_ApproximateMotion_from_graph(framework):
    motion1 = ApproximateMotion.from_graph(
        framework.graph,
        framework.realization(as_points=True, numerical=True),
        5,
        0.075,
    )
    for sample in motion1.motion_samples[1:]:
        assert framework.is_equivalent_realization(
            sample, numerical=True, tolerance=1e-3
        ) and not framework.is_congruent_realization(
            sample, numerical=True, tolerance=1e-3
        )

    try:
        motion2 = ApproximateMotion.from_graph(
            framework.graph,
            framework.realization(as_points=True, numerical=True),
            5,
            0.075,
            fixed_pair=(0, 1),
            fixed_direction=[1, 0],
        )
        for sample in motion2.motion_samples[1:]:
            assert framework.is_equivalent_realization(
                sample, numerical=True, tolerance=1e-3
            ) and not framework.is_congruent_realization(
                sample, numerical=True, tolerance=1e-3
            )
    except ValueError:
        assert framework._dim == 1


def test_normalize_realizations():
    F = fws.Path(3, dim=2)
    motion = ApproximateMotion(F, 10, 0.075, fixed_pair=(0, 1), fixed_direction=[1, 0])
    realizations = motion._normalize_realizations(motion.motion_samples, 2.02, 2.02)
    for r in realizations:
        assert (
            isclose(
                np.linalg.norm(
                    [
                        v - w
                        for v, w in zip(r[0], [0.5954750846569365, 0.15661445793796835])
                    ]
                ),
                0,
                abs_tol=1e-2,
            )
            and isclose(
                np.linalg.norm(
                    [
                        v - w
                        for v, w in zip(r[1], [2.0099998003349073, 0.15661445793796835])
                    ]
                ),
                0,
                abs_tol=1e-2,
            )
            and isclose(
                np.linalg.norm([v - w for v, w in zip(r[1], r[2])]),
                2,
                abs_tol=1e-2,
            )
            and np.linalg.norm(r[2][0]) <= 2.02
            and np.linalg.norm(r[2][1]) <= 2.02
        )


def test_ApproximateMotion_Getters():
    F = fws.Cycle(4)
    motion = ApproximateMotion(
        F,
        10,
        fixed_pair=(0, 1),
        fixed_direction=[0, 1],
        step_size=0.075,
        tolerance=1e-6,
        chosen_flex=0,
    )
    assert is_zero_vector(
        [
            (L - float(sp.sympify("sqrt(2)").evalf()))
            for L in motion.edge_lengths.values()
        ],
        numerical=True,
        tolerance=1e-6,
    )
    assert motion.steps == 10
    assert (
        len(motion.motion_samples) == 10
        and all(len(realization) == 4 for realization in motion.motion_samples)
        and all(
            F.is_equivalent_realization(realization, numerical=True, tolerance=1e-5)
            for realization in motion.motion_samples
        )
    )
    assert motion.tolerance == 1e-6
    assert F.is_congruent_realization(
        motion.starting_realization, numerical=True, tolerance=1e-5
    )
    assert motion.step_size == 0.075
    assert motion.fixed_pair == (0, 1)
    assert motion.fixed_direction == [0, 1]
    assert motion.pinned_vertex is None
    assert motion.chosen_flex == 0

    F = fws.Cycle(5)
    motion = ApproximateMotion(
        F, 15, pinned_vertex=1, step_size=0.05, tolerance=1e-5, chosen_flex=1
    )
    assert is_zero_vector(
        [
            L - float(sp.sympify("sqrt(2-2*cos(2*pi/5))").evalf())
            for L in motion.edge_lengths.values()
        ],
        numerical=True,
        tolerance=1e-6,
    )
    assert motion.steps == 15
    assert (
        len(motion.motion_samples) == 15
        and all(len(realization) == 5 for realization in motion.motion_samples)
        and all(
            F.is_equivalent_realization(realization, numerical=True, tolerance=1e-5)
            for realization in motion.motion_samples
        )
    )
    assert motion.tolerance == 1e-5
    assert F.is_congruent_realization(
        motion.starting_realization, numerical=True, tolerance=1e-5
    )
    assert motion.step_size == 0.05
    assert motion.fixed_pair is None
    assert motion.fixed_direction is None
    assert motion.pinned_vertex == 1
    assert motion.chosen_flex == 1


def test_fix_pair_of_vertices():
    F = fws.Cycle(4)
    motion = ApproximateMotion(
        F,
        10,
        fixed_pair=(0, 1),
        fixed_direction=[0, 1],
        step_size=0.075,
    )
    _motion = deepcopy(motion)
    motion.fix_pair_of_vertices((1, 2))
    assert all(
        F.is_equivalent_realization(realization, numerical=True, tolerance=1e-4)
        for realization in motion.motion_samples
    )
    for i, realization in enumerate(motion.motion_samples):
        _F = Framework(F.graph, realization)
        assert (
            _F.is_congruent_realization(
                _motion.motion_samples[i], numerical=True, tolerance=1e-4
            )
            and is_zero_vector(
                [
                    _motion.motion_samples[i][1][j]
                    - _motion.motion_samples[i][0][j]
                    - (float(sp.sympify("sqrt(2)").evalf()) if j == 0 else 0)
                    for j in range(0, 2)
                ],
                numerical=True,
                tolerance=1e-4,
            )
            and is_zero_vector(
                [
                    realization[2][j]
                    - realization[1][j]
                    - (float(sp.sympify("sqrt(2)").evalf()) if j == 0 else 0)
                    for j in range(0, 2)
                ],
                numerical=True,
                tolerance=1e-4,
            )
        )


def test_fix_vertex():
    F = fws.Cycle(6)
    motion = ApproximateMotion(
        F,
        10,
        pinned_vertex=0,
        step_size=0.075,
    )
    _motion = deepcopy(motion)
    motion.fix_vertex(2)
    assert all(
        F.is_equivalent_realization(realization, numerical=True, tolerance=1e-4)
        for realization in motion.motion_samples
    )
    for i, realization in enumerate(motion.motion_samples):
        _F = Framework(F.graph, realization)
        assert (
            _F.is_congruent_realization(
                _motion.motion_samples[i], numerical=True, tolerance=1e-4
            )
            and is_zero_vector(
                _motion.motion_samples[i][0],
                numerical=True,
                tolerance=1e-4,
            )
            and is_zero_vector(
                realization[2],
                numerical=True,
                tolerance=1e-4,
            )
        )
