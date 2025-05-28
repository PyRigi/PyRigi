"""
Module for the functionality concerning frameworks.
"""

from __future__ import annotations

from random import randrange
from typing import Any

import numpy as np
import sympy as sp
from sympy import Matrix

import pyrigi._utils._input_check as _input_check
from pyrigi._utils._doc import copy_doc, doc_category, generate_category_tables
from pyrigi.data_type import (
    DirectedEdge,
    Edge,
    InfFlex,
    Number,
    Point,
    Sequence,
    Stress,
    Vertex,
)
from pyrigi.framework.base import FrameworkBase
from pyrigi.graph import Graph
from pyrigi.graph import _general as graph_general
from pyrigi.graphDB import Complete as CompleteGraph
from pyrigi.plot_style import PlotStyle

from . import _general as general
from ._export import export
from ._plot import plot
from ._rigidity import infinitesimal as infinitesimal_rigidity
from ._rigidity import matroidal as matroidal_rigidity
from ._rigidity import redundant as redundant_rigidity
from ._rigidity import second_order as second_order_rigidity
from ._rigidity import stress as stress_rigidity
from ._transformations import transformations

__doctest_requires__ = {
    tuple(["Framework." + func_name for func_name in func_names]): pkgs
    for func_names, pkgs in export.__doctest_requires__.items()
}


class Framework(FrameworkBase):
    r"""
    This class provides the functionality for frameworks.

    Definitions
    -----------
     * :prf:ref:`Framework <def-framework>`
     * :prf:ref:`Realization <def-realization>`

    Parameters
    ----------
    graph:
        A graph without loops.
    realization:
        A dictionary mapping the vertices of the graph to points in $\RR^d$.
        The dimension $d$ is retrieved from the points in realization.
        If ``graph`` is empty, and hence also the ``realization``,
        the dimension is set to 0 (:meth:`.Empty`
        can be used to construct an empty framework with different dimension).

    Examples
    --------
    >>> F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
    >>> print(F)
    Framework in 2-dimensional space consisting of:
    Graph with vertices [0, 1] and edges [[0, 1]]
    Realization {0:(1, 2), 1:(0, 5)}

    Notice that the realization of a vertex can be accessed using ``[ ]``:

    >>> F[0]
    Matrix([
    [1],
    [2]])

    METHODS

    Notes
    -----
    Internally, the realization is represented as ``dict[Vertex,Matrix]``.
    However, :meth:`~Framework.realization` can also return ``dict[Vertex,Point]``.
    """

    @doc_category("Plotting")
    @copy_doc(plot.plot2D)
    def plot2D(
        self,
        plot_style: PlotStyle = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
        inf_flex: int | InfFlex = None,
        stress: int | Stress = None,
        edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        stress_label_positions: dict[DirectedEdge, float] = None,
        arc_angles_dict: Sequence[float] | dict[DirectedEdge, float] = None,
        filename: str = None,
        dpi=300,
        **kwargs,
    ) -> None:
        return plot.plot2D(
            self,
            plot_style=plot_style,
            projection_matrix=projection_matrix,
            random_seed=random_seed,
            coordinates=coordinates,
            inf_flex=inf_flex,
            stress=stress,
            edge_colors_custom=edge_colors_custom,
            stress_label_positions=stress_label_positions,
            arc_angles_dict=arc_angles_dict,
            filename=filename,
            dpi=dpi,
            **kwargs,
        )

    @doc_category("Plotting")
    @copy_doc(plot.animate3D_rotation)
    def animate3D_rotation(
        self,
        plot_style: PlotStyle = None,
        edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        total_frames: int = 100,
        delay: int = 75,
        rotation_axis: str | Sequence[Number] = None,
        **kwargs,
    ) -> Any:
        return plot.animate3D_rotation(
            self,
            plot_style=plot_style,
            edge_colors_custom=edge_colors_custom,
            total_frames=total_frames,
            delay=delay,
            rotation_axis=rotation_axis,
            **kwargs,
        )

    @doc_category("Plotting")
    @copy_doc(plot.plot3D)
    def plot3D(
        self,
        plot_style: PlotStyle = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
        inf_flex: int | InfFlex = None,
        stress: int | Stress = None,
        edge_colors_custom: Sequence[Sequence[Edge]] | dict[str, Sequence[Edge]] = None,
        stress_label_positions: dict[DirectedEdge, float] = None,
        filename: str = None,
        dpi=300,
        **kwargs,
    ) -> None:
        return plot.plot3D(
            self,
            plot_style=plot_style,
            projection_matrix=projection_matrix,
            random_seed=random_seed,
            coordinates=coordinates,
            inf_flex=inf_flex,
            stress=stress,
            edge_colors_custom=edge_colors_custom,
            stress_label_positions=stress_label_positions,
            filename=filename,
            dpi=dpi,
            **kwargs,
        )

    @doc_category("Plotting")
    @copy_doc(plot.plot)
    def plot(
        self,
        plot_style: PlotStyle = None,
        **kwargs,
    ) -> None:
        return plot.plot(
            self,
            plot_style=plot_style,
            **kwargs,
        )

    @doc_category("Other")
    @copy_doc(export.to_tikz)
    def to_tikz(
        self,
        vertex_style: str | dict[str, Sequence[Vertex]] = "fvertex",
        edge_style: str | dict[str, Sequence[Edge]] = "edge",
        label_style: str = "labelsty",
        figure_opts: str = "",
        vertex_in_labels: bool = False,
        vertex_out_labels: bool = False,
        default_styles: bool = True,
    ) -> str:
        return export.to_tikz(
            self,
            vertex_style=vertex_style,
            edge_style=edge_style,
            label_style=label_style,
            figure_opts=figure_opts,
            vertex_in_labels=vertex_in_labels,
            vertex_out_labels=vertex_out_labels,
            default_styles=default_styles,
        )

    @classmethod
    @doc_category("Class methods")
    def from_points(cls, points: Sequence[Point]) -> Framework:
        """
        Generate a framework from a list of points.

        The list of vertices of the underlying graph
        is taken to be ``[0,...,len(points)-1]``.
        The underlying graph has no edges.

        Parameters
        ----------
        points:
            The realization of the framework that this method outputs
            is provided as a list of points.

        Examples
        --------
        >>> F = Framework.from_points([(1,2), (2,3)])
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1] and edges []
        Realization {0:(1, 2), 1:(2, 3)}
        """
        vertices = range(len(points))
        realization = {v: points[v] for v in vertices}
        return Framework(Graph.from_vertices(vertices), realization)

    @classmethod
    @doc_category("Class methods")
    def Random(
        cls,
        graph: Graph,
        dim: int = 2,
        rand_range: int | Sequence[int] = None,
        numerical: bool = False,
    ) -> Framework:
        """
        Return a framework with random realization.

        Depending on the parameter ``numerical``, the realization either
        consists of random integers (``numerical=False``) or random floats
        (``numerical=True``).

        Parameters
        ----------
        dim:
            The dimension of the constructed framework.
        graph:
            Graph for which the random realization should be constructed.
        rand_range:
            Sets the range of random numbers from which the realization is
            sampled. The format is either an interval ``(a,b)`` or a single
            integer ``a``, which produces the range ``(-a,a)``.
            If ``rand_range=None``, then the range is set to ``(-a,a)`` for
            ``a = 10^4 * n * dim`` in the case that ``numerical=False``, where
            ``n`` is the number of vertices. For ``numerical=True``, we set the
            default interval to ``(-1,1)``.
        numerical:
            A boolean indicating whether numerical coordinates should be used.

        Examples
        --------
        >>> F = Framework.Random(Graph([(0,1), (1,2), (0,2)]))
        >>> print(F) # doctest: +SKIP
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(122, 57), 1:(27, 144), 2:(50, 98)}
        """
        _input_check.dimension(dim)
        if rand_range is None:
            if numerical:
                a, b = -1, 1
            else:
                b = 10**4 * graph.number_of_nodes() * dim
                a = -b
        elif isinstance(rand_range, list | tuple):
            if not len(rand_range) == 2:
                raise ValueError("If `rand_range` is a list, it must be of length 2.")
            a, b = rand_range
        elif isinstance(rand_range, int):
            if rand_range <= 0:
                raise ValueError("If `rand_range` is an int, it must be positive")
            b = rand_range
            a = -b
        else:
            raise TypeError("`rand_range` must be either a list or a single int.")

        if numerical:
            realization = {
                v: [a + np.random.rand() * (b - a) for _ in range(dim)]
                for v in graph.nodes
            }
        else:
            realization = {
                v: [randrange(a, b) for _ in range(dim)] for v in graph.nodes
            }

        return Framework(graph, realization)

    @classmethod
    @doc_category("Class methods")
    def Circular(cls, graph: Graph) -> Framework:
        """
        Return the framework with a regular unit circle realization in the plane.

        Parameters
        ----------
        graph:
            Underlying graph on which the framework is constructed.

        Examples
        ----
        >>> import pyrigi.graphDB as graphs
        >>> F = Framework.Circular(graphs.CompleteBipartite(4, 2))
        >>> print(F)
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2, 3, 4, 5] and edges ...
        Realization {0:(1, 0), 1:(1/2, sqrt(3)/2), ...
        """
        n = graph.number_of_nodes()
        return Framework(
            graph,
            {
                v: [sp.cos(2 * i * sp.pi / n), sp.sin(2 * i * sp.pi / n)]
                for i, v in enumerate(graph_general.vertex_list(graph))
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Collinear(cls, graph: Graph, dim: int = 1) -> Framework:
        """
        Return the framework with a realization on the x-axis.

        Parameters
        ----------
        dim:
            The dimension of the space in which the framework is constructed.
        graph:
            Underlying graph on which the framework is constructed.

        Examples
        --------
        >>> import pyrigi.graphDB as graphs
        >>> print(Framework.Collinear(graphs.Complete(3), dim=2))
        Framework in 2-dimensional space consisting of:
        Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]]
        Realization {0:(0, 0), 1:(1, 0), 2:(2, 0)}
        """
        _input_check.dimension(dim)
        return Framework(
            graph,
            {
                v: [i] + [0 for _ in range(dim - 1)]
                for i, v in enumerate(graph_general.vertex_list(graph))
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Simplicial(cls, graph: Graph, dim: int = None) -> Framework:
        """
        Return the framework with a realization on the ``dim``-simplex.

        Parameters
        ----------
        graph:
            Underlying graph on which the framework is constructed.
        dim:
            The dimension ``dim`` has to be at least the number of vertices
            of the ``graph`` minus one.
            If ``dim`` is not specified, then the least possible one is used.

        Examples
        ----
        >>> F = Framework.Simplicial(Graph([(0,1), (1,2), (2,3), (0,3)]), 4)
        >>> F.realization(as_points=True)
        {0: [0, 0, 0, 0], 1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0]}
        >>> F = Framework.Simplicial(Graph([(0,1), (1,2), (2,3), (0,3)]))
        >>> F.realization(as_points=True)
        {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
        """
        if dim is None:
            dim = graph.number_of_nodes() - 1
        _input_check.integrality_and_range(
            dim, "dimension d", max([1, graph.number_of_nodes() - 1])
        )
        return Framework(
            graph,
            {
                v: [1 if j == i - 1 else 0 for j in range(dim)]
                for i, v in enumerate(graph_general.vertex_list(graph))
            },
        )

    @classmethod
    @doc_category("Class methods")
    def Complete(cls, points: Sequence[Point]) -> Framework:
        """
        Generate a framework on the complete graph from a given list of points.

        The vertices of the underlying graph are taken
        to be the list ``[0,...,len(points)-1]``.

        Parameters
        ----------
        points:
            The realization of the framework that this method outputs
            is provided as a list of points.

        Examples
        --------
        >>> F = Framework.Complete([(1,),(2,),(3,),(4,)]); print(F)
        Framework in 1-dimensional space consisting of:
        Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        Realization {0:(1,), 1:(2,), 2:(3,), 3:(4,)}
        """  # noqa: E501
        if not points:
            raise ValueError("The list of points cannot be empty!")

        Kn = CompleteGraph(len(points))
        return Framework(Kn, {v: pos for v, pos in zip(Kn.nodes, points)})

    @doc_category("Framework properties")
    @copy_doc(general.is_quasi_injective)
    def is_quasi_injective(
        self, numerical: bool = False, tolerance: float = 1e-9
    ) -> bool:
        return general.is_quasi_injective(
            self, numerical=numerical, tolerance=tolerance
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_injective)
    def is_injective(self, numerical: bool = False, tolerance: float = 1e-9) -> bool:
        return general.is_injective(self, numerical=numerical, tolerance=tolerance)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.rigidity_matrix)
    def rigidity_matrix(
        self,
        vertex_order: Sequence[Vertex] = None,
        edge_order: Sequence[Edge] = None,
    ) -> Matrix:
        return infinitesimal_rigidity.rigidity_matrix(
            self, vertex_order=vertex_order, edge_order=edge_order
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.is_dict_stress)
    def is_dict_stress(self, dict_stress: dict[Edge, Number], **kwargs) -> bool:
        return stress_rigidity.is_dict_stress(self, dict_stress=dict_stress, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.is_vector_stress)
    def is_vector_stress(
        self,
        stress: Sequence[Number],
        edge_order: Sequence[Edge] = None,
        numerical: bool = False,
        tolerance=1e-9,
    ) -> bool:
        return stress_rigidity.is_vector_stress(
            self,
            stress=stress,
            edge_order=edge_order,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.is_stress)
    def is_stress(self, stress: Stress, **kwargs) -> bool:
        return stress_rigidity.is_stress(self, stress=stress, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.stress_matrix)
    def stress_matrix(
        self,
        stress: Stress,
        edge_order: Sequence[Edge] = None,
        vertex_order: Sequence[Vertex] = None,
    ) -> Matrix:
        return stress_rigidity.stress_matrix(
            self, stress=stress, edge_order=edge_order, vertex_order=vertex_order
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.trivial_inf_flexes)
    def trivial_inf_flexes(self, vertex_order: Sequence[Vertex] = None) -> list[Matrix]:
        return infinitesimal_rigidity.trivial_inf_flexes(
            self, vertex_order=vertex_order
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.nontrivial_inf_flexes)
    def nontrivial_inf_flexes(self, **kwargs) -> list[Matrix]:
        return infinitesimal_rigidity.nontrivial_inf_flexes(self, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.inf_flexes)
    def inf_flexes(
        self,
        include_trivial: bool = False,
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> list[Matrix] | list[list[float]]:
        return infinitesimal_rigidity.inf_flexes(
            self,
            include_trivial=include_trivial,
            vertex_order=vertex_order,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(stress_rigidity.stresses)
    def stresses(
        self,
        edge_order: Sequence[Edge] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> list[Matrix] | list[list[float]]:
        return stress_rigidity.stresses(
            self, edge_order=edge_order, numerical=numerical, tolerance=tolerance
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.rigidity_matrix_rank)
    def rigidity_matrix_rank(
        self, numerical: bool = False, tolerance: float = 1e-9
    ) -> int:
        return infinitesimal_rigidity.rigidity_matrix_rank(
            self, numerical=numerical, tolerance=tolerance
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_inf_rigid)
    def is_inf_rigid(self, numerical: bool = False, tolerance: float = 1e-9) -> bool:
        return infinitesimal_rigidity.is_inf_rigid(
            self, numerical=numerical, tolerance=tolerance
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_inf_flexible)
    def is_inf_flexible(self, **kwargs) -> bool:
        return infinitesimal_rigidity.is_inf_flexible(self, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_min_inf_rigid)
    def is_min_inf_rigid(self, use_copy: bool = True, **kwargs) -> bool:
        return infinitesimal_rigidity.is_min_inf_rigid(
            self, use_copy=use_copy, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(matroidal_rigidity.is_independent)
    def is_independent(self, **kwargs) -> bool:
        return matroidal_rigidity.is_independent(self, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(matroidal_rigidity.is_dependent)
    def is_dependent(self, **kwargs) -> bool:
        return matroidal_rigidity.is_dependent(self, **kwargs)

    @doc_category("Infinitesimal rigidity")
    @copy_doc(matroidal_rigidity.is_isostatic)
    def is_isostatic(self, **kwargs) -> bool:
        return matroidal_rigidity.is_isostatic(self, **kwargs)

    @doc_category("Other")
    @copy_doc(second_order_rigidity.is_prestress_stable)
    def is_prestress_stable(
        self,
        numerical: bool = False,
        tolerance: float = 1e-9,
        inf_flexes: Sequence[InfFlex] = None,
        stresses: Sequence[Stress] = None,
    ) -> bool:
        return second_order_rigidity.is_prestress_stable(
            self,
            numerical=numerical,
            tolerance=tolerance,
            inf_flexes=inf_flexes,
            stresses=stresses,
        )

    @doc_category("Other")
    @copy_doc(second_order_rigidity.is_second_order_rigid)
    def is_second_order_rigid(
        self,
        numerical: bool = False,
        tolerance: float = 1e-9,
        inf_flexes: Sequence[InfFlex] = None,
        stresses: Sequence[Stress] = None,
    ) -> bool:
        return second_order_rigidity.is_second_order_rigid(
            self,
            numerical=numerical,
            tolerance=tolerance,
            inf_flexes=inf_flexes,
            stresses=stresses,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(redundant_rigidity.is_redundantly_inf_rigid)
    def is_redundantly_inf_rigid(self, use_copy: bool = True, **kwargs) -> bool:
        return redundant_rigidity.is_redundantly_inf_rigid(
            self, use_copy=use_copy, **kwargs
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_congruent_realization)
    def is_congruent_realization(
        self,
        other_realization: dict[Vertex, Point] | dict[Vertex, Matrix],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return general.is_congruent_realization(
            self,
            other_realization=other_realization,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_congruent)
    def is_congruent(
        self,
        other_framework: Framework,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return general.is_congruent(
            self,
            other_framework=other_framework,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_equivalent_realization)
    def is_equivalent_realization(
        self,
        other_realization: dict[Vertex, Point] | dict[Vertex, Matrix],
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return general.is_equivalent_realization(
            self,
            other_realization=other_realization,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Framework properties")
    @copy_doc(general.is_equivalent)
    def is_equivalent(
        self,
        other_framework: Framework,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return general.is_equivalent(
            self,
            other_framework=other_framework,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Framework manipulation")
    @copy_doc(transformations.translate)
    def translate(
        self, vector: Point | Matrix, inplace: bool = True
    ) -> None | Framework:
        return transformations.translate(self, vector, inplace=inplace)

    @doc_category("Framework manipulation")
    @copy_doc(transformations.rescale)
    def rescale(self, factor: Number, inplace: bool = True) -> None | Framework:
        return transformations.rescale(self, factor, inplace=inplace)

    @doc_category("Framework manipulation")
    @copy_doc(transformations.rotate2D)
    def rotate2D(
        self, angle: float, rotation_center: Point = [0, 0], inplace: bool = True
    ) -> None | Framework:
        return transformations.rotate2D(
            self, angle, rotation_center=rotation_center, inplace=inplace
        )

    @doc_category("Framework manipulation")
    @copy_doc(transformations.rotate3D)
    def rotate3D(
        self,
        angle: Number,
        axis_direction: Sequence[Number] = [0, 0, 1],
        axis_shift: Point = [0, 0, 0],
        inplace: bool = True,
    ) -> None | Framework:
        return transformations.rotate3D(
            self,
            angle,
            axis_direction=axis_direction,
            axis_shift=axis_shift,
            inplace=inplace,
        )

    @doc_category("Framework manipulation")
    @copy_doc(transformations.rotate)
    def rotate(self, **kwargs) -> None | Framework:
        return transformations.rotate(self, **kwargs)

    @doc_category("Framework manipulation")
    @copy_doc(transformations.projected_realization)
    def projected_realization(
        self,
        proj_dim: int = None,
        projection_matrix: Matrix = None,
        random_seed: int = None,
        coordinates: Sequence[int] = None,
    ) -> tuple[dict[Vertex, Point], Matrix]:
        return transformations.projected_realization(
            self,
            proj_dim=proj_dim,
            projection_matrix=projection_matrix,
            random_seed=random_seed,
            coordinates=coordinates,
        )

    @doc_category("Other")
    @copy_doc(general.edge_lengths)
    def edge_lengths(self, numerical: bool = False) -> dict[Edge, Number]:
        return general.edge_lengths(self, numerical=numerical)

    @doc_category("Other")
    @copy_doc(export.generate_stl_bars)
    def generate_stl_bars(
        self,
        scale: float = 1.0,
        width_of_bars: float = 8.0,
        height_of_bars: float = 3.0,
        holes_diameter: float = 4.3,
        filename_prefix: str = "bar_",
        output_dir: str = "stl_output",
    ) -> None:
        return export.generate_stl_bars(
            self,
            scale=scale,
            width_of_bars=width_of_bars,
            height_of_bars=height_of_bars,
            holes_diameter=holes_diameter,
            filename_prefix=filename_prefix,
            output_dir=output_dir,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_vector_inf_flex)
    def is_vector_inf_flex(
        self,
        inf_flex: Sequence[Number],
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return infinitesimal_rigidity.is_vector_inf_flex(
            self,
            inf_flex=inf_flex,
            vertex_order=vertex_order,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_dict_inf_flex)
    def is_dict_inf_flex(
        self, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        return infinitesimal_rigidity.is_dict_inf_flex(
            self, vert_to_flex=vert_to_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_vector_nontrivial_inf_flex)
    def is_vector_nontrivial_inf_flex(
        self,
        inf_flex: Sequence[Number],
        vertex_order: Sequence[Vertex] = None,
        numerical: bool = False,
        tolerance: float = 1e-9,
    ) -> bool:
        return infinitesimal_rigidity.is_vector_nontrivial_inf_flex(
            self,
            inf_flex=inf_flex,
            vertex_order=vertex_order,
            numerical=numerical,
            tolerance=tolerance,
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_dict_nontrivial_inf_flex)
    def is_dict_nontrivial_inf_flex(
        self, vert_to_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        return infinitesimal_rigidity.is_dict_nontrivial_inf_flex(
            self, vert_to_flex=vert_to_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_nontrivial_flex)
    def is_nontrivial_flex(
        self,
        inf_flex: InfFlex,
        **kwargs,
    ) -> bool:
        return infinitesimal_rigidity.is_nontrivial_flex(
            self, inf_flex=inf_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_vector_trivial_inf_flex)
    def is_vector_trivial_inf_flex(self, inf_flex: Sequence[Number], **kwargs) -> bool:
        return infinitesimal_rigidity.is_vector_trivial_inf_flex(
            self, inf_flex=inf_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_dict_trivial_inf_flex)
    def is_dict_trivial_inf_flex(
        self, inf_flex: dict[Vertex, Sequence[Number]], **kwargs
    ) -> bool:
        return infinitesimal_rigidity.is_dict_trivial_inf_flex(
            self, inf_flex=inf_flex, **kwargs
        )

    @doc_category("Infinitesimal rigidity")
    @copy_doc(infinitesimal_rigidity.is_trivial_flex)
    def is_trivial_flex(
        self,
        inf_flex: InfFlex,
        **kwargs,
    ) -> bool:
        return infinitesimal_rigidity.is_trivial_flex(self, inf_flex=inf_flex, **kwargs)


Framework.__doc__ = Framework.__doc__.replace(
    "METHODS",
    generate_category_tables(
        Framework,
        1,
        [
            "Attribute getters",
            "Framework properties",
            "Class methods",
            "Framework manipulation",
            "Infinitesimal rigidity",
            "Plotting",
            "Other",
            "Waiting for implementation",
        ],
        include_all=False,
    ),
)
