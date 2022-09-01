import unittest

from blpytorchlightning.utils.spatial_derivative_kernels import (
    get_2d_first_derivative_kernels,
    get_2d_second_derivative_kernels
)


class TestSpatialDerivativeKernels(unittest.TestCase):
    """ Test that we can import the module and work with the functions. """

    def test_get_2d_first_derivative_kernels(self):
        k = get_2d_first_derivative_kernels()
        assert(isinstance(k, dict))

    def test_get_2d_first_derivative_kernels_contents(self):
        k = get_2d_first_derivative_kernels()
        assert("ddx" in k)
        assert("ddy" in k)

    def test_get_2d_second_derivative_kernels(self):
        k = get_2d_second_derivative_kernels()
        assert (isinstance(k, dict))

    def test_get_2d_second_derivative_kernels_contents(self):
        k = get_2d_second_derivative_kernels()
        assert("d2dx2" in k)
        assert("d2dy2" in k)
        assert("d2dxdy" in k)
