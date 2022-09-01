import unittest

from blpytorchlightning.dataset_components.PatchSampler import PatchSampler


class TestPatchSampler(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        PatchSampler()


if __name__ == '__main__':
    unittest.main()