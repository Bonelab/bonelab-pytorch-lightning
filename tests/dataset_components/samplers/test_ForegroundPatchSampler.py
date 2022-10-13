import unittest

from blpytorchlightning.dataset_components.samplers.ForegroundPatchSampler import ForegroundPatchSampler


class TestForegroundPatchSampler(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        ForegroundPatchSampler()


if __name__ == '__main__':
    unittest.main()
