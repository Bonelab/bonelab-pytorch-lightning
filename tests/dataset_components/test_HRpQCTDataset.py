import unittest

from blpytorchlightning.dataset_components.AIMLoader import AIMLoader
from blpytorchlightning.dataset_components.PatchSampler import PatchSampler
from blpytorchlightning.dataset_components.HRpQCTDataset import HRpQCTDataset


def get_file_loader():
    path = "."
    pattern = "*.AIM"
    return AIMLoader(path, pattern)


class TestHRpQCTDataset(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        HRpQCTDataset(get_file_loader(), PatchSampler())


if __name__ == '__main__':
    unittest.main()
