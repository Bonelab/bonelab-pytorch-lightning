import torch
import unittest

from blpytorchlightning.utils.error_metrics import dice_similarity_coefficient


class TestErrorMetrics(unittest.TestCase):
    """ Test that we can import the module and call the functions. """

    def test_dice_similarity_coefficient(self):
        y = torch.ones((3, 3))
        y_hat = torch.ones((3, 3))
        dice_similarity_coefficient(y, y_hat)
