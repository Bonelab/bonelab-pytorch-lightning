import unittest

from blpytorchlightning.dataset_components.samplers.MultisliceSampler import MultisliceSampler


class TestSliceSampler(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        MultisliceSampler()


if __name__ == '__main__':
    unittest.main()
