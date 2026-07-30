"""
This module provides algorithms related to redundant rigidity of frameworks.
"""

from __future__ import annotations

from copy import deepcopy

from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import _general as graph_general

from . import infinitesimal as infinitesimal_rigidity


def is_redundantly_inf_rigid(
    framework: FrameworkBase, use_copy: bool = True, **kwargs
) -> bool:
    """
    Return if the framework is infinitesimally redundantly rigid.

    For implementation details and possible parameters, see
    :func:`~.is_inf_rigid`.

    Definitions
    -----------
    :prf:ref:`Redundant infinitesimal rigidity <def-redundantly-rigid-framework>`

    Parameters
    ----------
    framework:
    use_copy:
        If ``False``, the framework's edges are deleted and added back
        during runtime.
        Otherwise, a new modified framework is created,
        while the original framework remains unchanged (default).

    Examples
    --------
    >>> F = Framework.Empty(dim=2)
    >>> F.add_vertices([(1,0), (1,1), (0,3), (-1,1)], ['a','b','c','d'])
    >>> F.add_edges([('a','b'), ('b','c'), ('c','d'), ('a','d'), ('a','c'), ('b','d')])
    >>> is_redundantly_inf_rigid(F)
    True
    >>> F.delete_edge(('a','c'))
    >>> is_redundantly_inf_rigid(F)
    False
    """  # noqa: E501
    F = framework
    if use_copy:
        F = deepcopy(framework)

    for edge in graph_general.edge_list(F._graph):
        F.delete_edge(edge)
        if not infinitesimal_rigidity.is_inf_rigid(F, **kwargs):
            F.add_edge(edge)
            return False
        F.add_edge(edge)
    return True
