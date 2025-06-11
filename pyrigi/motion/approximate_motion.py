"""
This module contains functionality related to approximations of motions.
"""

import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np

import pyrigi._utils._input_check as _input_check
from pyrigi._utils._conversion import sympy_expr_to_float
from pyrigi._utils._zero_check import is_zero, is_zero_vector
from pyrigi._utils.linear_algebra import _normalize_flex, _vector_distance_pointwise
from pyrigi.data_type import (
    DirectedEdge,
    Edge,
    InfFlex,
    Number,
    Point,
    Sequence,
    Vertex,
)
from pyrigi.framework import Framework
from pyrigi.framework._rigidity import infinitesimal as infinitesimal_rigidity
from pyrigi.graph import Graph
from pyrigi.graph import _general as graph_general
from pyrigi.motion.motion import Motion
from pyrigi.warning import NumericalAlgorithmWarning


class ApproximateMotion(Motion):
    """
    Class representing an approximated motion of a framework.

    When constructed, motion samples, i.e., realizations
    approximating a continuous flex of a given framework, are computed.

    Definitions
    -----------
    :prf:ref:`Continuous flex (motion)<def-motion>`

    Parameters
    ----------
    framework:
        A framework whose approximation of a continuous flex is computed.
    steps:
        The amount of retraction steps that are performed. This number is equal to the
        amount of motion samples that are computed.
    step_size:
        The step size of each retraction step. If the output seems too jumpy or instable,
        consider reducing the step size.
    chosen_flex:
        An integer indicating the ``i``-th flex from the list of :meth:`.Framework.inf_flexes`
        for ``i=chosen_flex``.
    tolerance:
        Tolerance for the Newton iteration.
    fixed_pair:
        Two vertices of the underlying graph that are fixed in each realization.
        By default, the first entry is pinned to the origin
        and the second is pinned to the ``x``-axis.
    fixed_direction:
        Vector to which the first direction is fixed. By default, this is given by
        the first and second entry of ``fixed_pair``.
    pinned_vertex:
        If the keyword ``fixed_pair`` is not set, we can use the keyword ``pinned_vertex``
        to pin one of the vertices to the origin instead during the motion.

    Examples
    --------
    >>> from pyrigi import graphDB as graphs
    >>> from pyrigi import ApproximateMotion
    >>> motion = ApproximateMotion.from_graph(
    ...     graphs.Cycle(4),
    ...     {0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)},
    ...     10
    ... )
    >>> print(motion)
    ApproximateMotion of a Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]] with starting configuration
    {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 1.0], 3: [0.0, 1.0]},
    10 retraction steps and initial step size 0.05.

    >>> F = Framework(graphs.Cycle(4), {0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)})
    >>> motion = ApproximateMotion(F, 10)
    >>> print(motion)
    ApproximateMotion of a Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]] with starting configuration
    {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 1.0], 3: [0.0, 1.0]},
    10 retraction steps and initial step size 0.05.
    """  # noqa: E501

    silence_numerical_alg_warns = False

    def __init__(
        self,
        framework: Framework,
        steps: int,
        step_size: float = 0.05,
        chosen_flex: int = 0,
        tolerance: float = 1e-9,
        fixed_pair: DirectedEdge = None,
        fixed_direction: Sequence[Number] = None,
        pinned_vertex: Vertex = None,
    ) -> None:
        """
        Create an instance of `ApproximateMotion`.
        """
        super().__init__(framework.graph, framework.dim)
        self._warn_numerical_alg(self.__init__)
        self._stress_length = len(framework.stresses())
        self._starting_realization = framework.realization(
            as_points=True, numerical=True
        )
        self._tolerance = tolerance
        self._steps = steps
        self._chosen_flex = chosen_flex
        self._step_size = step_size
        self._current_step_size = step_size
        self._edge_lengths = framework.edge_lengths(numerical=True)
        self._compute_motion_samples(chosen_flex)
        if fixed_pair is not None:
            _input_check.dimension_for_algorithm(
                self._dim, [2, 3], "ApproximateMotion._fix_edge"
            )
            if fixed_direction is None:
                fixed_direction = [1] + [0 for _ in range(self._dim - 1)]
            if len(fixed_direction) != self._dim:
                raise ValueError(
                    "`fixed_direction` does not have the same length as the"
                    + f" motion's dimension, which is {self._dim}."
                )
            self._motion_samples = self._fix_edge(
                self._motion_samples, fixed_pair, fixed_direction
            )
            self._pinned_vertex = None
        elif pinned_vertex is None:
            pin_result = self._pin_origin(self._motion_samples, pinned_vertex)
            self._motion_samples = pin_result[0]
            self._pinned_vertex = pin_result[1]
        else:
            if pinned_vertex not in self._graph.nodes:
                raise ValueError(
                    f"The pinned vertex {pinned_vertex} is not part of the graph"
                )
            self._motion_samples = self._pin_origin(
                self._motion_samples, pinned_vertex
            )[0]
            self._pinned_vertex = pinned_vertex
        self._fixed_pair = fixed_pair
        self._fixed_direction = fixed_direction

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        realization: dict[Vertex, Point],
        steps: int,
        step_size: float = 0.05,
        chosen_flex: int = 0,
        tolerance: float = 1e-9,
        fixed_pair: DirectedEdge = None,
        fixed_direction: Sequence[Number] = None,
        pinned_vertex: Vertex = None,
    ) -> Motion:
        """
        Create an instance from a graph with a realization.
        """
        if not len(realization) == graph.number_of_nodes():
            raise ValueError(
                "The realization does not contain the correct amount of vertices!"
            )

        realization = {v: sympy_expr_to_float(pos) for v, pos in realization.items()}
        realization_0 = realization[list(realization.keys())[0]]
        for v in graph.nodes:
            if v not in realization:
                raise KeyError(f"Vertex {v} is not a key of the given realization!")
            if len(realization[v]) != len(realization_0):
                raise ValueError(
                    f"The point {realization[v]} in the parametrization"
                    f" corresponding to vertex {v} does not have the right dimension."
                )
        F = Framework(graph, realization)
        return ApproximateMotion(
            F,
            steps,
            step_size=step_size,
            chosen_flex=chosen_flex,
            tolerance=tolerance,
            fixed_pair=fixed_pair,
            fixed_direction=fixed_direction,
            pinned_vertex=pinned_vertex,
        )

    def __str__(self) -> str:
        """Return the string representation."""
        res = super().__str__() + " with starting configuration\n"
        res += str(self.motion_samples[0]) + ",\n"
        res += str(self.steps) + " retraction steps and initial step size "
        res += str(self.step_size) + "."
        return res

    def __repr__(self) -> str:
        """Return a representation of the approximate motion."""
        o_str = f"ApproximateMotion.from_graph({repr(self._graph)}, "
        o_str += f"{self._starting_realization}, {self._steps}, "
        o_str += f"step_size={self._step_size}, chosen_flex={self._chosen_flex}, "
        o_str += f"tolerance={self._tolerance}, fixed_pair={self._fixed_pair}, "
        o_str += f"fixed_direction={self._fixed_direction}, "
        o_str += f"pinned_vertex={self._pinned_vertex})"
        return o_str

    @property
    def edge_lengths(self) -> dict[Edge, Number]:
        """
        Return a copy of the edge lengths.
        """
        return deepcopy(self._edge_lengths)

    @property
    def motion_samples(self) -> list[dict[Vertex, Point]]:
        """
        Return a copy of the motion samples.
        """
        return deepcopy(self._motion_samples)

    @property
    def steps(self) -> int:
        """
        Return the number of steps.
        """
        return self._steps

    @property
    def tolerance(self) -> float:
        """
        Return the tolerance for Newton's method.
        """
        return self._tolerance

    @property
    def starting_realization(self) -> dict[Vertex, Point]:
        """
        Return the starting realization of the motion.
        """
        return deepcopy(self._starting_realization)

    @property
    def step_size(self) -> float:
        """
        Return the step size of the motion.
        """
        return self._step_size

    @property
    def fixed_pair(self) -> DirectedEdge:
        """
        Return the fixed pair of the motion.
        """
        return deepcopy(self._fixed_pair)

    @property
    def chosen_flex(self) -> int:
        """
        Return the chosen flex of the motion.
        """
        return self._chosen_flex

    @property
    def pinned_vertex(self) -> Vertex:
        """
        Return the pinned vertex of the motion.
        """
        return self._pinned_vertex

    @property
    def fixed_direction(self) -> Sequence[Number]:
        """
        Return the vector to which `fixed_pair` is fixed in the motion.
        """
        return deepcopy(self._fixed_direction)

    def fix_vertex(self, vertex: Vertex) -> None:
        """
        Pin ``vertex`` to the origin.

        Parameters
        ----------
        vertex:
            The vertex that is pinned to the origin.
        """
        if vertex is None or vertex not in self.graph.nodes:
            raise ValueError(f"The pinned vertex {vertex} is not part of the graph")
        self._pinned_vertex = vertex
        self._motion_samples = self._pin_origin(self.motion_samples, vertex)[0]

    def fix_pair_of_vertices(
        self, fixed_pair: DirectedEdge, fixed_direction: Sequence[Number] = None
    ) -> None:
        """
        Pin ``fixed_pair`` to the vector given by ``fixed_direction``.

        The default value for ``fixed_direction`` is the first standard unit
        vector ``[1,0,...,0]``.

        Parameters
        ----------
        fixed_pair:
            The pair of vertices that is fixed in the direction ``fixed_direction``.
        fixed_direction:
            A vector to which the  pair of vertices is fixed.
        """
        if (
            len(fixed_pair) == 2
            and fixed_pair[0] in self.graph.nodes
            and fixed_pair[1] in self.graph.nodes
        ):
            _input_check.dimension_for_algorithm(
                self._dim, [2], "ApproximateMotion.fix_edge"
            )
            if fixed_direction is None:
                fixed_direction = [1] + [0 for _ in range(self._dim - 1)]
            if len(fixed_direction) != self._dim:
                raise ValueError(
                    "`fixed_direction` does not have the same length as the"
                    + f" motion's dimension, which is {self._dim}."
                )
        else:
            raise ValueError(
                "`fixed_pair` does not have the correct format or "
                + "has entries that are not contained in the underlying graph."
            )
        self._fixed_pair = fixed_pair
        self._fixed_direction = fixed_direction
        self._motion_samples = self._fix_edge(
            self.motion_samples, fixed_pair, fixed_direction
        )

    @classmethod
    def _warn_numerical_alg(cls, method: Callable) -> None:
        """
        Raise a warning if a numerical algorithm is silently called.

        Parameters
        ----------
        method:
            Reference to the method that is called.
        """
        if not cls.silence_numerical_alg_warns:
            warnings.warn(NumericalAlgorithmWarning(method, class_off=cls))

    def _compute_motion_samples(self, chosen_flex: int) -> None:
        """
        Perform path-tracking to compute the attribute ``_motion_samples``.
        """
        F = Framework(self._graph, self._starting_realization)
        inf_flexes = F.inf_flexes(numerical=True, tolerance=self.tolerance)
        _input_check.integrality_and_range(
            chosen_flex, "chosen_flex", max_val=len(inf_flexes)
        )
        cur_inf_flex = _normalize_flex(
            infinitesimal_rigidity._transform_inf_flex_to_pointwise(
                F, inf_flexes[chosen_flex]
            ),
            numerical=True,
        )

        cur_sol = self._starting_realization
        self._motion_samples = [cur_sol]
        i = 1
        # To avoid an infinite loop, the step size rescaling is reduced if only too large
        # or too small step sizes are found Its value converges to 1.
        step_size_rescaling = 2
        jump_indicator = [False, False]
        while i < self._steps:
            euler_step, cur_inf_flex = self._euler_step(cur_inf_flex, cur_sol)
            try:
                cur_sol = self._newton_steps(euler_step)
            except RuntimeError:
                # Try again with better discretization
                _discretization = 10
                self._current_step_size = self._current_step_size / _discretization
                for _ in range(_discretization):
                    try:
                        euler_step, cur_inf_flex = self._euler_step(
                            cur_inf_flex, cur_sol
                        )
                        cur_sol = self._newton_steps(euler_step)
                    except RuntimeError:
                        raise RuntimeError(
                            "Newton's method did not converge. Potentially the "
                            + "given framework is not flexible or the step size "
                            + "is too large?"
                        )
                self._current_step_size = self.step_size

            self._motion_samples += [cur_sol]
            # Reject the step if the step size is not close to what we expect
            if (
                _vector_distance_pointwise(
                    self._motion_samples[-1], self._motion_samples[-2], numerical=True
                )
                > self.step_size * 2
            ):
                self._current_step_size = self._current_step_size / step_size_rescaling
                self._motion_samples.pop()
                jump_indicator[0] = True
                if all(jump_indicator):
                    step_size_rescaling = step_size_rescaling ** (0.75)
                    jump_indicator = [False, False]
                continue
            elif (
                _vector_distance_pointwise(
                    self._motion_samples[-1], self._motion_samples[-2], numerical=True
                )
                < self.step_size / 2
            ):
                self._current_step_size = self._current_step_size * step_size_rescaling
                self._motion_samples.pop()
                jump_indicator[1] = True
                if all(jump_indicator):
                    step_size_rescaling = step_size_rescaling ** (0.75)
                    jump_indicator = [False, False]
                continue
            jump_indicator = [False, False]
            i = i + 1

    def _pin_origin(
        self, realizations: Sequence[dict[Vertex, Point]], pinned_vertex: Vertex = None
    ) -> tuple[list[dict[Vertex, Point]], Vertex]:
        """
        Pin a vertex to the origin.

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        pinned_vertex:
            Determines the vertex which is pinned to the origin.
        """
        _realizations = []
        if pinned_vertex is None:
            pinned_vertex = graph_general.vertex_list(self._graph)[0]
        for realization in realizations:
            if pinned_vertex not in realization.keys():
                raise ValueError(
                    "The `pinned_vertex` does not have a value in the provided motion."
                )

            # Translate the realization to the origin
            _realization = {
                v: [pos[i] - realization[pinned_vertex][i] for i in range(len(pos))]
                for v, pos in realization.items()
            }
            _realizations.append(_realization)
        return _realizations, pinned_vertex

    @staticmethod
    def _fix_edge(
        realizations: Sequence[dict[Vertex, Point]],
        fixed_pair: DirectedEdge,
        fixed_direction: Sequence[Number],
    ) -> list[dict[Vertex, Point]]:
        """
        Fix the two vertices in ``fixed_pair`` for every entry of ``realizations``.

        Parameters
        ----------
        realizations:
            A list of realization samples describing the motion.
        fixed_pair:
            Two vertices of the underlying graph that should not move during
            the animation. By default, the first entry is pinned to the origin
            and the second is pinned to the `x`-axis.
        fixed_direction:
            Vector to which the first direction is fixed. By default, this is given by
            the first and second entry.
        """
        if len(fixed_pair) != 2:
            raise TypeError("The length of `fixed_pair` is not 2.")
        (v1, v2) = (fixed_pair[0], fixed_pair[1])
        if not (v1 in realizations[0] and v2 in realizations[0]):
            raise ValueError(
                "The vertices of the edge {realizations} are not part of the graph."
            )

        # Translate the realization to the origin
        _realizations = [
            {
                v: [pos[i] - realization[v1][i] for i in range(len(pos))]
                for v, pos in realization.items()
            }
            for realization in realizations
        ]
        if fixed_direction is None:
            fixed_direction = [
                x - y for x, y in zip(_realizations[0][v1], _realizations[0][v2])
            ]
            if is_zero(np.linalg.norm(fixed_direction), numerical=True, tolerance=1e-6):
                warnings.warn(
                    f"The entries of the edge {fixed_pair} are too close to each "
                    + "other. Thus, `fixed_direction=(1,0)` is chosen instead."
                )
                fixed_direction = [1] + [
                    0
                    for _ in range(
                        len(_realizations[list(_realizations.keys())[0]]) - 1
                    )
                ]
            else:
                fixed_direction = [
                    coord / np.linalg.norm(fixed_direction) for coord in fixed_direction
                ]

        output_realizations = []
        for realization in _realizations:
            if any([len(pos) not in [2, 3] for pos in realization.values()]):
                raise ValueError(
                    "This method ``_fix_edge`` is not implemented for "
                    + "dimensions other than 2 or 3."
                )
            if (
                len(fixed_direction) not in [2, 3]
                or np.linalg.norm(fixed_direction) != 1
            ):
                raise ValueError("`fixed_direction` does not have the correct format.")

            # Compute the signed angle `theta` between the `fixed_direction` and the
            # vector `realization[v2]`
            if len(fixed_direction) == 2:
                theta = np.arctan2(
                    [fixed_direction[1], realization[v2][1]],
                    [fixed_direction[0], realization[v2][0]],
                )[1]

                rotation_matrix = np.array(
                    [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
                )

            else:
                edge_vector = [
                    pos / np.linalg.norm(realization[v2]) for pos in realization[v2]
                ]
                rotation_axis = np.cross(fixed_direction, edge_vector)
                if is_zero_vector(rotation_axis, numerical=True):
                    rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                else:
                    rotation_axis = [
                        pos / np.linalg.norm(rotation_axis) for pos in rotation_axis
                    ]
                    angle = np.acos(np.dot(fixed_direction, edge_vector))
                    rotation_matrix = np.array(
                        [
                            [
                                np.cos(angle)
                                + rotation_axis[0] ** 2 * (1 - np.cos(angle)),
                                rotation_axis[0]
                                * rotation_axis[1]
                                * (1 - np.cos(angle))
                                - rotation_axis[2] * np.sin(angle),
                                rotation_axis[0]
                                * rotation_axis[2]
                                * (1 - np.cos(angle))
                                + rotation_axis[1] * np.sin(angle),
                            ],
                            [
                                rotation_axis[0]
                                * rotation_axis[1]
                                * (1 - np.cos(angle))
                                + rotation_axis[2] * np.sin(angle),
                                np.cos(angle)
                                + rotation_axis[1] ** 2 * (1 - np.cos(angle)),
                                rotation_axis[1]
                                * rotation_axis[2]
                                * (1 - np.cos(angle))
                                - rotation_axis[0] * np.sin(angle),
                            ],
                            [
                                rotation_axis[0]
                                * rotation_axis[2]
                                * (1 - np.cos(angle))
                                - rotation_axis[1] * np.sin(angle),
                                rotation_axis[1]
                                * rotation_axis[2]
                                * (1 - np.cos(angle))
                                + rotation_axis[0] * np.sin(angle),
                                np.cos(angle)
                                + rotation_axis[2] ** 2 * (1 - np.cos(angle)),
                            ],
                        ]
                    )
                    rotation_matrix = np.linalg.inv(rotation_matrix)
            # Rotate the realization to the `fixed_direction`.
            _realization = {
                v: np.dot(rotation_matrix, pos) for v, pos in realization.items()
            }
            output_realizations.append(_realization)
        return output_realizations

    def animate(
        self,
        **kwargs,
    ) -> Any:
        """
        Animate the approximate motion.

        See the parent method :meth:`~.Motion.animate` for the list of possible keywords.
        """
        realizations = self._motion_samples
        return super().animate(
            realizations,
            None,
            **kwargs,
        )

    def _euler_step(
        self,
        old_inf_flex: InfFlex,
        realization: dict[Vertex, Point],
    ) -> tuple[dict[Vertex, Point], InfFlex]:
        """
        Compute a single Euler step.

        This method returns the resulting configuration and the infinitesimal flex
        that was used in the computation as a tuple.

        Notes
        -----
        Choose the (normalized) infinitesimal flex with the smallest distance from the
        previous infinitesimal flex ``old_inf_flex``. This is given by computing the
        Moore-Penrose pseudoinverse.

        Suggested Improvements
        ----------------------
        * Add vector transport to ``old_inf_flex`` to more accurately compare the vectors.
        * Search the space of `inf_flexes` using a Least Squares approach rather than
        just searching a basis
        """
        F = Framework(self._graph, realization)

        inf_flex_space = np.vstack(
            F.inf_flexes(numerical=True, tolerance=self.tolerance)
        )
        old_inf_flex_matrix = np.reshape(
            sum([list(pos) for pos in old_inf_flex.values()], []), (-1, 1)
        )
        flex_coefficients = np.dot(
            np.linalg.pinv(inf_flex_space).transpose(), old_inf_flex_matrix
        )
        predicted_inf_flex = sum(
            np.dot(inf_flex_space.transpose(), flex_coefficients).tolist(), []
        )
        predicted_inf_flex = _normalize_flex(
            infinitesimal_rigidity._transform_inf_flex_to_pointwise(
                F, predicted_inf_flex
            ),
            numerical=True,
        )
        realization = self._motion_samples[-1]
        return {
            v: tuple(
                [
                    pos[i] + self._current_step_size * predicted_inf_flex[v][i]
                    for i in range(len(realization[v]))
                ]
            )
            for v, pos in realization.items()
        }, predicted_inf_flex

    def _newton_steps(self, realization: dict[Vertex, Point]) -> dict[Vertex, Point]:
        """
        Compute a sequence of Newton steps to return to the constraint variety.

        Notes
        -----
        There are more robust implementations of Newton's method (using damped schemes and
        preconditioning), but that would blow method out of the current scope. Here, a
        naive damping scheme is implemented (so that the method is actually guaranteed
        to converge), but this means that in the case where dim(stresses=flexes), the
        damping goes to 0. MH has tested this so-called "damped Gauß-Newton scheme"
        extensively in two other packages. If the equations are randomized first, there
        are convergence and smoothness guarantees from numerical algebraic geometry,
        but that is currently out of the scope of the `ApproximateMotion` class.

        Suggested Improvements
        ----------------------
        Randomize the bar-length equations.
        """
        F = Framework(self._graph, realization)
        cur_sol = np.array(
            sum(
                [list(realization[v]) for v in graph_general.vertex_list(self._graph)],
                [],
            )
        )
        equations = [
            np.linalg.norm(
                [
                    x - y
                    for x, y in zip(
                        cur_sol[(self._dim * e[0]) : (self._dim * (e[0] + 1))],
                        cur_sol[(self._dim * e[1]) : (self._dim * (e[1] + 1))],
                    )
                ]
            )
            ** 2
            - length**2
            for e, length in self._edge_lengths.items()
        ]
        cur_error = prev_error = np.linalg.norm(equations)
        damping = 0.2
        rand_mat = np.random.rand(
            F._graph.number_of_edges() - self._stress_length, F._graph.number_of_edges()
        )
        while cur_error > self.tolerance:
            rigidity_matrix = np.array(
                F.rigidity_matrix(edge_order=self._edge_lengths.keys())
            ).astype(np.float64)
            if self._stress_length > 0:
                equations = np.dot(rand_mat, equations)
                rigidity_matrix = np.dot(rand_mat, rigidity_matrix)
            newton_step = np.dot(np.linalg.pinv(rigidity_matrix), equations)

            cur_sol = [
                cur_sol[i] - 0.5 * damping * newton_step[i] for i in range(len(cur_sol))
            ]
            equations = [
                np.linalg.norm(
                    [
                        x - y
                        for x, y in zip(
                            cur_sol[(self._dim * e[0]) : (self._dim * (e[0] + 1))],
                            cur_sol[(self._dim * e[1]) : (self._dim * (e[1] + 1))],
                        )
                    ]
                )
                ** 2
                - length**2
                for e, length in self._edge_lengths.items()
            ]
            cur_error = np.linalg.norm(equations)
            if cur_error < prev_error:
                damping = damping * 1.2
            else:
                damping = damping / 2
            # If the damping becomes too small, raise an exception.

            if damping < 1e-14:
                raise RuntimeError("Newton's method did not converge.")
            prev_error = cur_error

        return {
            v: tuple(cur_sol[(self._dim * i) : (self._dim * (i + 1))])
            for i, v in enumerate(graph_general.vertex_list(self._graph))
        }
