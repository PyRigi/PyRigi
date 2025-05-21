"""
This module provides algorithms related to rigidity matroids of frameworks.
"""

from pyrigi.framework.base import FrameworkBase

from . import infinitesimal as infinitesimal_rigidity


def is_independent(framework: FrameworkBase, **kwargs) -> bool:
    """
    Return whether the framework is independent.

    For implementation details and possible parameters, see
    :meth:`~Framework.rigidity_matrix_rank`.

    Definitions
    -----------
    :prf:ref:`Independent framework <def-independent-framework>`

    Examples
    --------
    >>> F = Framework.Complete([[0,0], [1,0], [1,1], [0,1]])
    >>> F.is_independent()
    False
    >>> F.delete_edge((0,2))
    >>> F.is_independent()
    True
    """
    return (
        infinitesimal_rigidity.rigidity_matrix_rank(framework, **kwargs)
        == framework._graph.number_of_edges()
    )


def is_dependent(framework: FrameworkBase, **kwargs) -> bool:
    """
    Return whether the framework is dependent.

    See also :meth:`~.Framework.is_independent`.

    Definitions
    -----------
    :prf:ref:`Dependent framework <def-independent-framework>`
    """
    return not is_independent(framework, **kwargs)


def is_isostatic(framework: FrameworkBase, **kwargs) -> bool:
    """
    Return whether the framework is isostatic.

    For implementation details and possible parameters, see
    :meth:`~Framework.is_independent` and
    :meth:`~Framework.is_inf_rigid`.

    Definitions
    -----------
    :prf:ref:`Isostatic framework <def-isostatic-frameworks>`
    """
    return is_independent(framework, **kwargs) and infinitesimal_rigidity.is_inf_rigid(
        framework, **kwargs
    )
