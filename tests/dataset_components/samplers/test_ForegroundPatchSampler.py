import unittest
import numpy as np

from blpytorchlightning.dataset_components.samplers.ForegroundPatchSampler import ForegroundPatchSampler


class TestForegroundPatchSampler(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        ForegroundPatchSampler()

    def test_can_sample(self):
        sampler = ForegroundPatchSampler()
        sample = (np.zeros((1, 500, 500)), np.zeros((1, 500, 500)))
        sample = sampler(sample)


if __name__ == '__main__':
    unittest.main()
