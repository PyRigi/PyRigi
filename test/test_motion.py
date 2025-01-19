from pyrigi.motion import ParametricMotion, ApproximateMotion
import pyrigi.graphDB as graphs
import pyrigi.frameworkDB as fws
import sympy as sp
import numpy as np
import pytest


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
    R = mot.realization(0, numeric=False)
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

    R = mot.realization("2/3", numeric=False)
    tmp = R[2]
    assert tmp[0] == sp.sympify("-7/5")
    assert tmp[1] == sp.sympify("9/5")

    tmp = R[3]
    assert tmp[0] == sp.sympify("-16/65")
    assert tmp[1] == sp.sympify("-63/65")

    R = mot.realization(2 / 3, numeric=True)
    tmp = R[2]
    assert abs(tmp[0] - (-7 / 5)) < 1e-9
    assert abs(tmp[1] - 9 / 5) < 1e-9

    tmp = R[3]
    assert abs(tmp[0] - (-16 / 65)) < 1e-9
    assert abs(tmp[1] - (-63 / 65)) < 1e-9


def test_ParametricMotion_init():
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


def test_ApproximateMotion_init():
    ApproximateMotion.from_framework(fws.Cycle(4), 100)


@pytest.mark.slow_main
def test_animate():
    F = fws.Square()
    M = ApproximateMotion(F.graph(), F.realization(as_points=True, numerical=True), 118, 0.075)
    assert np.linalg.norm([u-v for u,v in zip(sum([list(val) for val in M.motion_samples[0].values()],[]), sum([list(val) for val in M.motion_samples[-1].values()],[]))])<1e-2
    M.animate()
