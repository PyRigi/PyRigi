from math import isclose

import sympy as sp
import numpy as np
import pytest

import pyrigi.frameworkDB as fws
import pyrigi.graphDB as graphs
from pyrigi import Framework, ParametricMotion, ApproximateMotion, Graph


def test_check_edge_lengths():
    mot = ParametricMotion(
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
    assert mot.check_edge_lengths()

    t = sp.Symbol("t")
    mot = ParametricMotion(
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
    assert mot.check_edge_lengths()

    mot = {
        0: ("t", "0"),
        1: ("1", "0"),
        2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
        3: (
            "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
            "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
        ),
    }
    with pytest.raises(ValueError):
        ParametricMotion(graphs.Cycle(4), mot, [-sp.oo, sp.oo])

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

    mot = ParametricMotion(graphs.CompleteBipartite(4, 4), p, [-sp.pi, sp.pi])
    assert mot.check_edge_lengths()


def test_realization():
    mot = ParametricMotion(
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
    R = mot.realization(0, numerical=False)
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

    R = mot.realization("2/3", numerical=False)
    tmp = R[2]
    assert tmp[0] == sp.sympify("-7/5")
    assert tmp[1] == sp.sympify("9/5")

    tmp = R[3]
    assert tmp[0] == sp.sympify("-16/65")
    assert tmp[1] == sp.sympify("-63/65")

    R = mot.realization(2 / 3, numerical=True)
    tmp = R[2]
    assert abs(tmp[0] - (-7 / 5)) < 1e-9
    assert abs(tmp[1] - 9 / 5) < 1e-9

    tmp = R[3]
    assert abs(tmp[0] - (-16 / 65)) < 1e-9
    assert abs(tmp[1] - (-63 / 65)) < 1e-9


def test_ParametricMotion_init():
    mot = {
        0: [
            "t",
        ],
        1: [
            "t",
        ],
    }
    mot = ParametricMotion(graphs.Path(2), mot, [-10, 10])
    mot.animate(animation_format="svg")
    mot.animate(animation_format="matplotlib")

    mot = {
        0: ("k", "0"),
        1: ("1", "0"),
        2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
        3: (
            "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
            "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
        ),
    }
    with pytest.raises(ValueError):
        ParametricMotion(graphs.Cycle(4), mot, [-sp.oo, sp.oo])

    mot = {
        7: ("0", "0"),
        1: ("1", "0"),
        2: ("4 * (t**2 - 2) / (t**2 + 4)", "12 * t / (t**2 + 4)"),
        3: (
            "(t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4)",
            "6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4)",
        ),
    }
    with pytest.raises(KeyError):
        ParametricMotion(graphs.Cycle(4), mot, [-sp.oo, sp.oo])

    t = 0
    mot = {
        0: (0, 0),
        1: (1, 0),
        2: (4 * (t**2 - 2) / (t**2 + 4), 12 * t / (t**2 + 4)),
        3: (
            (t**4 - 13 * t**2 + 4) / (t**4 + 5 * t**2 + 4),
            6 * (t**3 - 2 * t) / (t**4 + 5 * t**2 + 4),
        ),
    }

    with pytest.raises(ValueError):
        ParametricMotion(graphs.Cycle(4), mot, [-sp.oo, sp.oo])


@pytest.mark.parametrize(
    "F",
    [
        Framework(Graph([(0, 1), (2, 3)]), {0: [0], 1: [1], 2: [2], 3: [3]}),
        fws.Square(),
        fws.Cycle(5),
        fws.Cycle(6),
        fws.ThreePrism("flexible"),
        fws.CompleteBipartite(2, 4),
        pytest.param(fws.CompleteBipartite(2, 5), marks=pytest.mark.slow_main),
    ],
)
def test_animate(F):
    M = ApproximateMotion(F, 5, 0.075)
    M.animate(animation_format="svg")
    M.animate(animation_format="matplotlib")

    # 1D Framework
    G = Graph([(0, 1), (2, 3)])
    F = Framework(G, {0: [0], 1: [1], 2: [2], 3: [3]})
    M = ApproximateMotion(F, 50, 0.1)
    for sample in M.motion_samples[1:]:
        assert F.is_equivalent_realization(
            sample, numerical=True, tolerance=1e-3
        ) and not F.is_congruent_realization(sample, numerical=True)


def test_animate3D():
    F = fws.Cube()
    M = ApproximateMotion(F, 5, 0.075)
    for sample in M.motion_samples[1:]:
        assert F.is_equivalent_realization(
            sample, numerical=True, tolerance=1e-3
        ) and not F.is_congruent_realization(sample, numerical=True)
    M.animate()


@pytest.mark.parametrize(
    "F",
    [
        Framework(Graph([(0, 1), (2, 3)]), {0: [0], 1: [1], 2: [2], 3: [3]}),
        fws.Square(),
        fws.Cycle(5),
        fws.Cycle(6),
        fws.ThreePrism("flexible"),
        fws.CompleteBipartite(2, 4),
        pytest.param(fws.CompleteBipartite(2, 5), marks=pytest.mark.slow_main),
    ],
)
def test_ApproximateMotion_from_framework(F):
    M1 = ApproximateMotion(F, 5, 0.075)
    for sample in M1.motion_samples[1:]:
        assert F.is_equivalent_realization(
            sample, numerical=True, tolerance=1e-3
        ) and not F.is_congruent_realization(sample, numerical=True)

    try:
        M2 = ApproximateMotion(F, 5, 0.075, fixed_pair=(0, 1), fixed_direction=[1, 0])
        for sample in M2.motion_samples[1:]:
            assert F.is_equivalent_realization(
                sample, numerical=True, tolerance=1e-3
            ) and not F.is_congruent_realization(sample, numerical=True)
    except ValueError:
        assert F._dim == 1


@pytest.mark.parametrize(
    "F",
    [
        Framework(Graph([(0, 1), (2, 3)]), {0: [0], 1: [1], 2: [2], 3: [3]}),
        fws.Square(),
        fws.Cycle(5),
        fws.Cycle(6),
        fws.ThreePrism("flexible"),
        fws.CompleteBipartite(2, 4),
        pytest.param(fws.CompleteBipartite(2, 5), marks=pytest.mark.slow_main),
    ],
)
def test_ApproximateMotion_from_graph(F):
    M1 = ApproximateMotion.from_graph(
        F.graph(), F.realization(as_points=True, numerical=True), 5, 0.075
    )
    for sample in M1.motion_samples[1:]:
        assert F.is_equivalent_realization(
            sample, numerical=True, tolerance=1e-3
        ) and not F.is_congruent_realization(sample, numerical=True)

    try:
        M2 = ApproximateMotion.from_graph(
            F.graph(),
            F.realization(as_points=True, numerical=True),
            5,
            0.075,
            fixed_pair=(0, 1),
            fixed_direction=[1, 0],
        )
        for sample in M2.motion_samples[1:]:
            assert F.is_equivalent_realization(
                sample, numerical=True, tolerance=1e-3
            ) and not F.is_congruent_realization(sample, numerical=True)
    except ValueError:
        assert F._dim == 1


def test_normalize_realizations():
    F = fws.Path(3, dim=2)
    M = ApproximateMotion(F, 10, 0.075, fixed_pair=(0, 1), fixed_direction=[1, 0])
    realizations = M._normalize_realizations(M.motion_samples, 2.02, 2.02)
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


@pytest.mark.long_local
def test_newton_raises_runtimeerror():
    F = fws.ThreePrism(realization="flexible")
    with pytest.raises(RuntimeError):
        ApproximateMotion(F, 5, 0.1)
