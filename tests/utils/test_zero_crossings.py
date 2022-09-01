import torch
import unittest

from blpytorchlightning.utils.zero_crossings import zero_crossings


class TestZeroCrossings(unittest.TestCase):
    """ Test that we can import the module and use the functions. """

    def test_call_zero_crossings(self):
        zero_crossings(torch.zeros(1, 1, 5, 5))

    def test_zero_crossings_line(self):
        x = torch.arange(-3, 4) * torch.ones((1, 1, 7, 7))
        zc_hat = torch.Tensor([False, False, True, False, False]) * torch.ones((1, 1, 5, 5))
        zc = zero_crossings(x)
        assert(torch.all(zc == zc_hat))
