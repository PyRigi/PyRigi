"""
This module holds functions related to questions whether
a graph can have a :prf:ref:`NAC-coloring <def-nac>`.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx

from pyrigi.graph._flexibility.nac.core import NACColoring
from pyrigi.graph._flexibility.nac.existence import (
    _can_have_NAC_coloring,
    _check_for_vertex_out_of_3_cycle,
    check_NAC_constrains,
    has_NAC_coloring_checks,
)
from pyrigi.graph._flexibility.nac.mono_classes import MonoClassType
from pyrigi.graph._flexibility.nac.search import NAC_colorings_impl


def _single_general_NAC_coloring(
    graph: nx.Graph,
    algorithm: str | Literal["naive", "subgraphs"],
    use_cycles_optimization: bool,
    mono_class_type: MonoClassType,
    seed: int | None,
) -> NACColoring | None:
    """
    Find a :prf:ref:`NAC-coloring <def-nac>` of a graph if possible.

    Before running the full extensive search,
    graph connectivity checks are used first to determine if there exists
    a trivial :prf:ref:`NAC-coloring <def-nac>`.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`

    Parameters
    ----------
    graph:
        The graph to find the :prf:ref:`NAC-coloring <def-nac>` for.

    Return
    ------
    A NAC-coloring certificate if a NAC-coloring exists, None otherwise.
    """
    components: list[set[int]] = list(
        nx.algorithms.components.connected_components(graph)
    )

    if len(components) > 1:
        # filter all the single nodes
        components = list(filter(lambda nodes: len(nodes) > 1, components))

        # there are more disconnected components with at least one edge,
        # we can color both of them with different color and be done.
        if len(components) > 1:
            red, blue = set(), set()
            for u, v in nx.edges(graph):
                (red if u in components[0] else blue).add((u, v))
            return (red, blue)

        # if there is only one component with all the edges,
        # the NAC coloring exists <=> this component has NAC coloring
        return single_NAC_coloring_impl(
            nx.Graph(graph.subgraph(components[0])),
            algorithm=algorithm,
            use_cycles_optimization=use_cycles_optimization,
            mono_class_type=mono_class_type,
            seed=seed,
        )

    if nx.algorithms.connectivity.node_connectivity(graph) < 2:
        generator = nx.algorithms.biconnected_components(graph)
        component: set[int] = next(generator)
        assert next(generator)  # make sure there are more components

        red, blue = set(), set()
        for v, u in graph.edges:
            (red if v in component and u in component else blue).add((u, v))

        return (red, blue)

    return None


def has_NAC_coloring_impl(
    graph: nx.Graph,
    algorithm: str | Literal["naive", "subgraphs"],
    use_cycles_optimization: bool,
    mono_class_type: MonoClassType,
    seed: int | None,
) -> bool:
    """
    Check whether the graph has a :prf:ref:`NAC-coloring <def-nac>`.

    Same as :func:`pyrigi.graph._flexibility.nac.single.single_NAC_coloring_impl`,
    but the certificate may not be created.
    Polynomial time checks are used.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`

    Parameters
    ----------
    graph:
        The graph to check.

    Return
    ------
    True if the graph has a :prf:ref:`NAC-coloring <def-nac>`, False otherwise.
    """
    if not check_NAC_constrains(graph):
        return False

    res = has_NAC_coloring_checks(graph)
    if res is not None:
        return res

    return (
        single_NAC_coloring_impl(
            graph,
            algorithm=algorithm,
            use_cycles_optimization=use_cycles_optimization,
            mono_class_type=mono_class_type,
            seed=seed,
            # we already checked some things
            _is_first_check=False,
        )
        is not None
    )


def single_NAC_coloring_impl(
    graph: nx.Graph,
    algorithm: str | Literal["naive", "subgraphs"],
    use_cycles_optimization: bool,
    mono_class_type: MonoClassType,
    seed: int | None,
    _is_first_check: bool = True,
) -> NACColoring | None:
    """
    Find a single :prf:ref:`NAC-coloring <def-nac>` if it exists.

    Polynomial time existence checks are run to determine
    whether a NAC-coloring exists.
    If they fail, an exhaustive search is run
    and halted after some NAC-coloring is found.

    Definitions
    -----------
    * :prf:ref:`NAC-coloring <def-nac>`

    Parameters
    ----------
    algorithm:
        The algorithm used in case we need to fall back
        to exhaustive search.
    use_cycles_optimization:
        Use cycles optimization for the given algorithm.
        This is always enabled for subgraphs strategies.
    mono_class_type:
        The type of :prf:ref:`NAC-mono classes <def-nac-mono>` to use.
        The options are ``"edges"`` (each edge is a NAC-mono class),
        ``"triangle"``
        for :prf:ref:`triangle-connected components <def-triangle-connected-comp>`,
        or ``"triangle-extended"`` (default) for
    seed:
        The seed to use in case we need to fall back to exhaustive search.
    _is_first_check:
        Internal parameter, do not change!
        Skips some already checked checks in has_NAC_coloring.
    ----------
    """
    if _is_first_check:
        if not check_NAC_constrains(graph):
            return None

        res = _check_for_vertex_out_of_3_cycle(graph)
        if res is not None:
            return res

        res = _single_general_NAC_coloring(
            graph,
            algorithm=algorithm,
            use_cycles_optimization=use_cycles_optimization,
            mono_class_type=mono_class_type,
            seed=seed,
        )
        if res is not None:
            return res

        # Need to be run after connectivity checks
        if not _can_have_NAC_coloring(graph):
            return None

    return next(
        iter(
            NAC_colorings_impl(
                graph=graph,
                algorithm=algorithm,
                use_cycles_optimization=use_cycles_optimization,
                # we already checked for bridges
                use_blocks_decomposition=False,
                mono_class_type=mono_class_type,
                seed=seed,
            )
        ),
        None,
    )
