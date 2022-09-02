import unittest

from blpytorchlightning.loss_functions.CurvatureLoss import CurvatureLoss


class TestCurvatureLoss(unittest.TestCase):
    """ Test that we can import the module and work with the file """

    def test_can_instantiate(self):
        vox_width = 1.0
        curvature_threshold = 0.001
        CurvatureLoss(vox_width, curvature_threshold)
