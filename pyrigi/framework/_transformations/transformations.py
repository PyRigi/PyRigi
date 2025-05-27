"""
This module provides transformations of frameworks.
"""

from copy import deepcopy

import numpy as np
import sympy as sp
from sympy import Matrix

import pyrigi._utils._input_check as _input_check
from pyrigi._utils._conversion import point_to_vector
from pyrigi._utils._zero_check import is_zero_vector
from pyrigi._utils.linear_algebra import (
    _generate_three_orthonormal_vectors,
    _generate_two_orthonormal_vectors,
)
from pyrigi.data_type import (
    Number,
    Point,
    Sequence,
    Vertex,
)
from pyrigi.framework.base import FrameworkBase


def translate(
    framework: FrameworkBase, vector: Point | Matrix, inplace: bool = True
) -> None | FrameworkBase:
    """
    Translate the framework.

    Parameters
    ----------
    vector
        Translation vector
    inplace
        If ``True`` (default), then this framework is translated.
        Otherwise, a new translated framework is returned.
    """
    vector = point_to_vector(vector)

    if inplace:
        if vector.shape[0] != framework.dim:
            raise ValueError(
                "The dimension of the vector has to be the same as of the framework!"
            )

        for v in framework._realization.keys():
            framework._realization[v] += vector
        return

    new_framework = deepcopy(framework)
    translate(new_framework, vector, True)
    return new_framework


def rescale(
    framework: FrameworkBase, factor: Number, inplace: bool = True
) -> None | FrameworkBase:
    """
    Scale the framework.

    Parameters
    ----------
    factor:
        Scaling factor
    inplace:
        If ``True`` (default), then this framework is translated.
        Otherwise, a new translated framework is returned.
    """
    if isinstance(factor, str):
        factor = sp.sympify(factor)
    if inplace:
        for v in framework._realization.keys():
            framework._realization[v] = framework._realization[v] * factor
        return

    new_framework = deepcopy(framework)
    rescale(new_framework, factor, True)
    return new_framework


def rotate2D(
    framework: FrameworkBase,
    angle: float,
    rotation_center: Point = [0, 0],
    inplace: bool = True,
) -> None | FrameworkBase:
    """
    Rotate the planar framework counterclockwise.

    Parameters
    ----------
    angle:
        Rotation angle. If you want the coordinates to stay symbolic
        even after the rotation, you need to input the angle as a
        `sympy` expression.
    rotation_center:
        The center of rotation.
        By default, this is the origin.
    inplace:
        If ``True`` (default), then this framework is rotated.
        Otherwise, a new rotated framework is returned.
    """
    _input_check.dimension_for_algorithm(framework.dim, [2], "rotate2D")
    _input_check.equal(len(rotation_center), 2, "length of the `rotation_center`")

    rotation_matrix = Matrix(
        [[sp.cos(angle), -sp.sin(angle)], [sp.sin(angle), sp.cos(angle)]]
    )

    opposite_rotation_center = [-rotation_center[0], -rotation_center[1]]

    if inplace:
        rotated_framework = framework
    else:
        rotated_framework = deepcopy(framework)

    translate(rotated_framework, opposite_rotation_center, inplace=True)
    for v, pos in rotated_framework._realization.items():
        rotated_framework._realization[v] = rotation_matrix * pos
    translate(rotated_framework, rotation_center, inplace=True)

    if inplace:
        return
    else:
        return rotated_framework


def rotate3D(
    framework: FrameworkBase,
    angle: Number,
    axis_direction: Sequence[Number] = [0, 0, 1],
    axis_shift: Point = [0, 0, 0],
    inplace: bool = True,
) -> None | FrameworkBase:
    """
    Rotate the spatial framework counterclockwise around a specified rotation axis.

    Parameters
    ----------
    angle:
        Rotation angle around the axis of rotation. If you want the
        coordinates to stay symbolic even after the rotation, you need
        to input the angle as a `sympy` expression.
    axis_direction:
        Direction of the rotation axis.
        By default, this is the ``z``-axis.
    axis_shift:
        A point through which the rotation axis passes.
        By default, this is the origin.
    inplace:
        If ``True`` (default), then this framework is rotated.
        Otherwise, a new rotated framework is returned.
    """
    _input_check.dimension_for_algorithm(framework.dim, [3], "rotate3D")
    _input_check.equal(len(axis_direction), 3, "length of the `axis_direction`")
    _input_check.equal(len(axis_shift), 3, "length of the `axis_shift`")
    if is_zero_vector(axis_direction):
        raise ValueError(
            "The parameter `axis_direction` needs to be a non-zero vector."
        )

    versor_dir_axis = [
        pos / sp.sqrt(sum(coord**2 for coord in axis_direction))
        for pos in axis_direction
    ]
    rotation_matrix = Matrix(
        [
            [
                sp.cos(angle) + versor_dir_axis[0] ** 2 * (1 - sp.cos(angle)),
                versor_dir_axis[0] * versor_dir_axis[1] * (1 - sp.cos(angle))
                - versor_dir_axis[2] * sp.sin(angle),
                versor_dir_axis[0] * versor_dir_axis[2] * (1 - sp.cos(angle))
                + versor_dir_axis[1] * sp.sin(angle),
            ],
            [
                versor_dir_axis[0] * versor_dir_axis[1] * (1 - sp.cos(angle))
                + versor_dir_axis[2] * sp.sin(angle),
                sp.cos(angle) + versor_dir_axis[1] ** 2 * (1 - sp.cos(angle)),
                versor_dir_axis[1] * versor_dir_axis[2] * (1 - sp.cos(angle))
                - versor_dir_axis[0] * sp.sin(angle),
            ],
            [
                versor_dir_axis[0] * versor_dir_axis[2] * (1 - sp.cos(angle))
                - versor_dir_axis[1] * sp.sin(angle),
                versor_dir_axis[1] * versor_dir_axis[2] * (1 - sp.cos(angle))
                + versor_dir_axis[0] * sp.sin(angle),
                sp.cos(angle) + versor_dir_axis[2] ** 2 * (1 - sp.cos(angle)),
            ],
        ]
    )

    opposite_axis_shift = [-axis_shift[0], -axis_shift[1], -axis_shift[2]]

    if inplace:
        rotated_framework = framework
    else:
        rotated_framework = deepcopy(framework)

    translate(rotated_framework, opposite_axis_shift, inplace=True)
    for v, pos in rotated_framework._realization.items():
        rotated_framework._realization[v] = rotation_matrix * pos
    translate(rotated_framework, axis_shift, inplace=True)

    if inplace:
        return
    else:
        return rotated_framework


def rotate(framework: FrameworkBase, **kwargs) -> None | FrameworkBase:
    """
    Alias for rotating frameworks based on
    :meth:`~Framework.rotate2D` and :meth:`~Framework.rotate3D`.

    For implementation details and possible parameters, see
    :meth:`~Framework.rotate2D` and :meth:`~Framework.rotate3D`.
    """
    _input_check.dimension_for_algorithm(framework.dim, [2, 3], "rotate")
    if framework.dim == 2:
        return rotate2D(framework, **kwargs)
    elif framework.dim == 3:
        return rotate3D(framework, **kwargs)


def projected_realization(
    framework: FrameworkBase,
    proj_dim: int = None,
    projection_matrix: Matrix = None,
    random_seed: int = None,
    coordinates: Sequence[int] = None,
) -> tuple[dict[Vertex, Point], Matrix]:
    """
    Return the realization projected to a lower dimension and the projection matrix.

    Parameters
    ----------
    proj_dim:
        The dimension to which the framework is projected.
        This is determined from ``projection_matrix`` if it is provided.
    projection_matrix:
        The matrix used for projecting the placement of vertices.
        The matrix must have dimensions ``(proj_dim, dim)``,
        where ``dim`` is the dimension of the given framework.
        If ``None``, a numerical random projection matrix is generated.
    random_seed:
        The random seed used for generating the projection matrix.
    coordinates:
        Indices of coordinates to which projection is applied.
        Providing the parameter overrides the previous ones.

    Suggested Improvements
    ----------------------
    Generate random projection matrix over symbolic rationals.
    """
    if coordinates is not None:
        if not isinstance(coordinates, tuple) and not isinstance(coordinates, list):
            raise TypeError("The parameter ``coordinates`` must be a tuple or a list.")
        if max(coordinates) >= framework.dim:
            raise ValueError(
                f"Index {np.max(coordinates)} out of range"
                + f" with placement in dim: {framework.dim}."
            )
        if isinstance(proj_dim, int) and len(coordinates) != proj_dim:
            raise ValueError(
                f"The number of coordinates ({len(coordinates)}) does not match"
                + f" proj_dim ({proj_dim})."
            )
        matrix = np.zeros((len(coordinates), framework.dim))
        for i, coord in enumerate(coordinates):
            matrix[i, coord] = 1

        return (
            {
                v: tuple([pos[coord] for coord in coordinates])
                for v, pos in framework._realization.items()
            },
            Matrix(matrix),
        )

    if projection_matrix is not None:
        projection_matrix = np.array(projection_matrix)
        if projection_matrix.shape[1] != framework.dim:
            raise ValueError(
                "The projection matrix has wrong number of columns."
                + f"{projection_matrix.shape[1]} instead of {framework.dim}."
            )
        if isinstance(proj_dim, int) and projection_matrix.shape[0] != proj_dim:
            raise ValueError(
                "The projection matrix has wrong number of rows."
                + f"{projection_matrix.shape[0]} instead of {framework.dim}."
            )

    if projection_matrix is None:
        if proj_dim == 2:
            projection_matrix = _generate_two_orthonormal_vectors(
                framework.dim, random_seed=random_seed
            )
        elif proj_dim == 3:
            projection_matrix = _generate_three_orthonormal_vectors(
                framework.dim, random_seed=random_seed
            )
        else:
            raise ValueError(
                "An automatically generated random matrix is supported"
                + f" only in dimension 2 or 3. {proj_dim} was given instead."
            )
        projection_matrix = projection_matrix.T
    return (
        {
            v: tuple([float(s[0]) for s in np.dot(projection_matrix, np.array(pos))])
            for v, pos in framework.realization(as_points=False, numerical=True).items()
        },
        projection_matrix,
    )
