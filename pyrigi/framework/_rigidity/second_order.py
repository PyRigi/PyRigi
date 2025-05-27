"""
This module provides algorithms related to second order rigidity of frameworks.
"""

import sympy as sp
from sympy import Matrix

from pyrigi._utils._conversion import sympy_expr_to_float
from pyrigi._utils._zero_check import is_zero
from pyrigi.data_type import (
    Edge,
    InfFlex,
    Number,
    Point,
    Sequence,
    Stress,
    Vertex,
)
from pyrigi.framework._rigidity import infinitesimal as infinitesimal_rigidity
from pyrigi.framework._rigidity import stress as stress_rigidity
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import _general as graph_general


def is_prestress_stable(
    framework: FrameworkBase,
    numerical: bool = False,
    tolerance: float = 1e-9,
    inf_flexes: Sequence[InfFlex] = None,
    stresses: Sequence[Stress] = None,
) -> bool:
    """
    Return whether the framework is prestress stable.

    See also :meth:`.is_second_order_rigid`.

    Definitions
    ----------
    :prf:ref:`Prestress stability <def-prestress-stability>`

    Parameters
    -------
    numerical:
        If ``True``, numerical infinitesimal flexes and stresses
        are used in the check for prestress stability.
        In case that ``numerical=False``, this method only
        properly works for symbolic coordinates.
    tolerance:
        Numerical tolerance used for the check that something is
        an approximate zero.
    inf_flexes, stresses:
        Precomputed infinitesimal flexes and equilibrium stresses can be provided
        to avoid recomputation. If not provided, they are computed here.

    Examples
    --------
    >>> from pyrigi import frameworkDB as fws
    >>> F = fws.Frustum(3)
    >>> F.is_prestress_stable()
    True
    """
    edges = graph_general.edge_list(framework._graph, as_tuples=True)
    inf_flexes = _process_list_of_inf_flexes(
        framework, inf_flexes, numerical=numerical, tolerance=tolerance
    )
    if len(inf_flexes) == 0:
        return True
    stresses = _process_list_of_stresses(
        framework, stresses, numerical=numerical, tolerance=tolerance
    )
    if len(stresses) == 0:
        return False

    if len(inf_flexes) == 1:
        q = inf_flexes[0]
        stress_energy_list = []
        for stress in stresses:
            stress_energy_list.append(
                sum(
                    [
                        stress[(u, v)]
                        * sum(
                            [
                                (q1 - q2) ** 2
                                for q1, q2 in zip(
                                    q[u],
                                    q[v],
                                )
                            ]
                        )
                        for u, v in edges
                    ]
                )
            )
        return any(
            [
                not is_zero(Q, numerical=numerical, tolerance=tolerance)
                for Q in stress_energy_list
            ]
        )

    if len(stresses) == 1:
        a = sp.symbols("a0:%s" % len(inf_flexes), real=True)
        stress_energy = 0
        stress_energy += sum(
            [
                stresses[0][(u, v)]
                * sum(
                    [
                        (
                            sum(
                                [
                                    a[i] * (inf_flexes[i][u][j] - inf_flexes[i][v][j])
                                    for i in range(len(inf_flexes))
                                ]
                            )
                            ** 2
                        )
                        for j in range(framework.dim)
                    ]
                )
                for u, v in edges
            ]
        )

        coefficients = {
            (i, j): sp.Poly(stress_energy, a).coeff_monomial(a[i] * a[j])
            for i in range(len(inf_flexes))
            for j in range(i, len(inf_flexes))
        }
        #  We then apply the SONC criterion.
        if numerical:
            return all(
                [
                    coefficients[(i, j)] ** 2
                    < sympy_expr_to_float(
                        4 * coefficients[(i, i)] * coefficients[(j, j)]
                    )
                    for i in range(len(inf_flexes))
                    for j in range(i + 1, len(inf_flexes))
                ]
            )
        sonc_expressions = [
            sp.simplify(
                sp.cancel(
                    4 * coefficients[(i, i)] * coefficients[(j, j)]
                    - coefficients[(i, j)] ** 2
                )
            )
            for i in range(len(inf_flexes))
            for j in range(i + 1, len(inf_flexes))
        ]
        if any(expr is None for expr in sonc_expressions):
            raise RuntimeError(
                "It could not be determined by `sympy.simplify` "
                + "whether the given sympy expression can be simplified."
                + "Please report this as an issue on Github "
                + "(https://github.com/PyRigi/PyRigi/issues)."
            )
        sonc_expressions = [expr.is_positive for expr in sonc_expressions]
        if any(expr is None for expr in sonc_expressions):
            raise RuntimeError(
                "It could not be determined by `sympy.is_positive` "
                + "whether the given sympy expression is positive."
                + "Please report this as an issue on Github "
                + "(https://github.com/PyRigi/PyRigi/issues)."
            )
        return all(sonc_expressions)

    raise ValueError("Prestress stability is not yet implemented for the general case.")


def is_second_order_rigid(
    framework: FrameworkBase,
    numerical: bool = False,
    tolerance: float = 1e-9,
    inf_flexes: Sequence[InfFlex] = None,
    stresses: Sequence[Stress] = None,
) -> bool:
    """
    Return whether the framework is second-order rigid.

    Checking second-order-rigidity for a general framework is computationally hard.
    If there is only one stress or only one infinitesimal flex, second-order rigidity
    is identical to :prf:ref:`prestress stability <def-prestress-stability>`,
    so we can apply :meth:`.is_prestress_stable`. See also
    :prf:ref:`thm-second-order-implies-prestress-stability`.

    Definitions
    ----------
    :prf:ref:`Second-order rigidity <def-second-order-rigid>`

    Parameters
    -------
    numerical:
        If ``True``, numerical infinitesimal flexes and stresses
        are used in the check for prestress stability.
        In case that ``numerical=False``, this method only
        properly works for symbolic coordinates.
    tolerance:
        Numerical tolerance used for the check that something is
        an approximate zero.
    inf_flexes, stresses:
        Precomputed infinitesimal flexes and equilibrium stresses can be provided
        to avoid recomputation. If not provided, they are computed here.

    Examples
    --------
    >>> from pyrigi import frameworkDB as fws
    >>> F = fws.Frustum(3)
    >>> F.is_second_order_rigid()
    True
    """
    inf_flexes = _process_list_of_inf_flexes(
        framework, inf_flexes, numerical=numerical, tolerance=tolerance
    )
    if len(inf_flexes) == 0:
        return True
    stresses = _process_list_of_stresses(
        framework, stresses, numerical=numerical, tolerance=tolerance
    )
    if len(stresses) == 0:
        return False

    if len(stresses) == 1 or len(inf_flexes) == 1:
        return is_prestress_stable(
            framework,
            numerical=numerical,
            tolerance=tolerance,
            inf_flexes=inf_flexes,
            stresses=stresses,
        )

    raise ValueError("Second-order rigidity is not implemented for this framework.")


def _process_list_of_inf_flexes(
    framework: FrameworkBase,
    inf_flexes: Sequence[InfFlex],
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> list[dict[Vertex, Point]]:
    """
    Process the input infinitesimal flexes for the second-order methods.

    If any of the input is not a nontrivial flex, an error is thrown.
    Otherwise, the infinitesimal flexes are transformed to a ``list`` of
    ``dict``.

    Parameters
    ----------
    inf_flexes:
        The infinitesimal flexes to be processed.
    numerical:
        If ``True``, the check is numerical.
    tolerance:
        Numerical tolerance used for the check that something is
        a nontrivial infinitesimal flex.
    """
    if inf_flexes is None:
        inf_flexes = infinitesimal_rigidity.inf_flexes(
            framework, numerical=numerical, tolerance=tolerance
        )
        if len(inf_flexes) == 0:
            return inf_flexes
    elif any(
        not infinitesimal_rigidity.is_nontrivial_flex(
            framework, inf_flex, numerical=numerical, tolerance=tolerance
        )
        for inf_flex in inf_flexes
    ):
        raise ValueError(
            "Some of the provided `inf_flexes` are not "
            + "nontrivial infinitesimal flexes!"
        )
    if len(inf_flexes) == 0:
        raise ValueError("No infinitesimal flexes were provided.")
    if all(isinstance(inf_flex, list | tuple | Matrix) for inf_flex in inf_flexes):
        inf_flexes = [
            infinitesimal_rigidity._transform_inf_flex_to_pointwise(framework, q)
            for q in inf_flexes
        ]
    elif not all(isinstance(inf_flex, dict) for inf_flex in inf_flexes):
        raise ValueError("The provided `inf_flexes` do not have the correct format.")
    return inf_flexes


def _process_list_of_stresses(
    framework: FrameworkBase,
    stresses: Sequence[Stress],
    numerical: bool = False,
    tolerance: float = 1e-9,
) -> list[dict[Edge, Number]]:
    """
    Process the input equilibrium stresses for the second-order methods.

    If any of the input is not an equilibrium stress, an error is thrown.
    Otherwise, the equilibrium stresses are transformed to a list of
    ``dict``.

    Parameters
    ----------
    stresses:
        The equilibrium stresses to be processed.
    numerical:
        If ``True``, the check is numerical.
    tolerance:
        Numerical tolerance used for the check that something is
        an equilibrium stress.
    """
    edges = graph_general.edge_list(framework._graph, as_tuples=True)
    if stresses is None:
        stresses = stress_rigidity.stresses(
            framework, numerical=numerical, tolerance=tolerance
        )
        if len(stresses) == 0:
            return stresses
    elif any(
        not stress_rigidity.is_stress(
            framework, stress, numerical=numerical, tolerance=tolerance
        )
        for stress in stresses
    ):
        raise ValueError(
            "Some of the provided `stresses` are not equilibrium stresses!"
        )
    if len(stresses) == 0:
        raise ValueError("No equilibrium stresses were provided.")
    if all(isinstance(stress, list | tuple | Matrix) for stress in stresses):
        stresses = [
            stress_rigidity._transform_stress_to_edgewise(
                framework, stress, edge_order=edges
            )
            for stress in stresses
        ]
    elif not all(isinstance(stress, dict) for stress in stresses):
        raise ValueError("The provided `stresses` do not have the correct format.")
    return stresses
