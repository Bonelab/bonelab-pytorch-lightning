import unittest

from blpytorchlightning.dataset_components.samplers.SlicePatchSampler import SlicePatchSampler


class TestSlicePatchSampler(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        SlicePatchSampler()


if __name__ == '__main__':
    unittest.main()
