import unittest

from blpytorchlightning.dataset_components.samplers.SliceSampler import SliceSampler


class TestSliceSampler(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        SliceSampler()


if __name__ == '__main__':
    unittest.main()
