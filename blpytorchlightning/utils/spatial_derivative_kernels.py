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
