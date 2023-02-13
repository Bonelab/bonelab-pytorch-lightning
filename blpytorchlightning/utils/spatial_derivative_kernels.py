from __future__ import annotations

import numpy as np


def get_2d_first_derivative_kernels() -> dict[np.ndarray]:
    """
    Create a dictionary of 2D stencils for computing second-order first spatial derivatives of a scalar field.
    For example, can be used to find the gradient.

    Returns
    -------
    dict[np.ndarray]
        Contains the fields `ddx` and `ddy` for the stencils for the x- and y-component of the gradient, respectively.
    """
    return {
        "ddx": np.array([[0, 0, 0], [-1 / 2, 0, 1 / 2], [0, 0, 0]]),
        "ddy": np.array([[0, -1 / 2, 0], [0, 0, 0], [0, 1 / 2, 0]]),
    }


def get_2d_second_derivative_kernels() -> dict[np.ndarray]:
    """
    Create a dictionary of 2D stencils for computing second-order second spatial derivatives of a scalar field.
    For example, can be used to find the hessian.

    Returns
    -------
    dict[np.ndarray]
        Contains the fields `d2dx2`, d2dy2`, and `d2dxdy` for the second derivative with respect to x and y, and the
        mixed second derivative, respectively.
    """
    return {
        "d2dx2": np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]),
        "d2dy2": np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]),
        "d2dxdy": np.array([[1 / 4, 0, -1 / 4], [0, 0, 0], [-1 / 4, 0, 1 / 4]]),
    }


def get_3d_first_derivative_kernels(prewitt: bool = True) -> dict[np.ndarray]:
    """
    Create a dictionary of 3D stencils for computing second-order first spatial derivatives of a scalar field.
    For example, can be used to find the gradient.

    Parameters
    ----------
    prewitt : bool
        Flag that controls whether to use the prewitt operator, if false use a normal kernel

    Returns
    -------
    dict[np.ndarray]
        Contains the fields `ddx`, `ddy`, and 'ddz' for the stencils for the x- and y-component of the gradient, respectively.
    """
    base_kernel = np.zeros((3, 3, 3))
    kernels = {}
    for i, label in enumerate(['ddx', 'ddy', 'ddz']):
        kernels[label] = base_kernel.copy()
        kernels[label][tuple([0 if j == i else slice(None) if prewitt else 1 for j in range(3)])] = 1 / 2
        kernels[label][tuple([2 if j == i else slice(None) if prewitt else 1 for j in range(3)])] = -1 / 2
    return kernels


def get_3d_second_derivative_kernels() -> dict[np.ndarray]:
    """
    Create a dictionary of 3D stencils for computing second-order second spatial derivatives of a scalar field.
    For example, can be used to find the hessian.

    Returns
    -------
    dict[np.ndarray]
        Contains the fields `d2dx2`, d2dy2`, 'd2dz2', 'd2dxdy', 'd2dydz', 'd2dxdz'
    """
    base_kernel = np.zeros((3, 3, 3))
    kernels = {}
    for i, label in enumerate(['d2dx2', 'd2dy2', 'd2dz2']):
        kernels[label] = base_kernel.copy()
        kernels[label][tuple([0 if j == i else slice(None) if prewitt else 1 for j in range(3)])] = 1
        kernels[label][tuple([1 if j == i else slice(None) if prewitt else 1 for j in range(3)])] = -2
        kernels[label][tuple([2 if j == i else slice(None) if prewitt else 1 for j in range(3)])] = 1
    for (i, j), label in zip([(0, 1), (1, 2), (0, 2)], ['d2dxdy', 'd2dydz', 'd2dxdz']):
        kernels[label] = base_kernel.copy()
        kernels[label][tuple([0 if (k == i or k == j) else 1 for k in range(3)])] = 1 / 4
        kernels[label][tuple([2 if (k == i or k == j) else 1 for k in range(3)])] = 1 / 4
        kernels[label][tuple([2 if k == i else 0 if k == j else 1 for k in range(3)])] = -1 / 4
        kernels[label][tuple([0 if k == i else 2 if k == j else 1 for k in range(3)])] = -1 / 4
    return kernels
