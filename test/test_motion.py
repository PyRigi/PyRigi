from pyrigi.motion import ParametricMotion
from pyrigi.graph import Graph
import pyrigi.graphDB as graphs
import sympy as sp
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
        mot = ParametricMotion(graphs.Cycle(4), mot, [-sp.oo, sp.oo])


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
    assert tmp[0] == 0

    tmp = R[1]
    assert tmp[0] == 1
    assert tmp[1] == 0

    tmp = R[2]
    assert tmp[0] == -2
    assert tmp[1] == 0

    tmp = R[3]
    assert tmp[0] == 1
    assert tmp[1] == 0


def test_parmotion_init():
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
        tmp = ParametricMotion(graphs.Cycle(4), mot, [-sp.oo, sp.oo])

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
        mot = ParametricMotion(graphs.Cycle(4), mot, [-sp.oo, sp.oo])

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
        tmp = ParametricMotion(graphs.Cycle(4), mot, [-sp.oo, sp.oo])
