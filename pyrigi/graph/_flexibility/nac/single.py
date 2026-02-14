"""
This module holds functions related to questions whether a graph has a NAC coloring.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx

from pyrigi.graph._flexibility.nac.core import NACColoring
from pyrigi.graph._flexibility.nac.existence import (
    _can_have_flexible_labeling,
    _check_for_simple_stable_cut,
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
    Find a trivial NAC coloring if possible.

    The algorithm is based on connectivity of the graph components.
    Returned coloring is trivially both NAC coloring.
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
    Same as :func:`pyrigi.graph._flexibility.nac.single.single_NAC_coloring_impl`, but the certificate may not be created,
    so some additional tricks are used the performance may be improved.
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
    Finds only a single NAC coloring if it exists.

    Parameters
    ----------
    algorithm:
        The algorithm used in case we need to fall back
        to exhaustive search.
    _is_first_check:
        Internal parameter, do not change!
        Skips some already checked checks in has_NAC_coloring.
    ----------
    """
    if _is_first_check:
        if not check_NAC_constrains(graph):
            return None

        res = _check_for_simple_stable_cut(graph)
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
        if not _can_have_flexible_labeling(graph):
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
