import unittest

from blpytorchlightning.loss_functions.MagnitudeGradientSDTLoss import MagnitudeGradientSDTLoss


class TestMagnitudeGradientSDTLoss(unittest.TestCase):
    """ Test that we can import the module and work with the file """

    def test_can_instantiate(self):
        vox_width = 1.0
        MagnitudeGradientSDTLoss(vox_width)
