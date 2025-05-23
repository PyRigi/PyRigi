from copy import deepcopy
from math import isclose

import numpy as np
import pytest
import sympy as sp

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi import ApproximateMotion, Framework, Graph
from pyrigi._utils._zero_check import is_zero_vector


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
def test_from_framework(framework):
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
def test_from_graph(framework):
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


def test_getters():
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
