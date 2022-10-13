import unittest

from blpytorchlightning.dataset_components.file_loaders.AIMLoader import AIMLoader
from blpytorchlightning.dataset_components.samplers.PatchSampler import PatchSampler
from blpytorchlightning.dataset_components.datasets.ComposedDataset import ComposedDataset


def get_file_loader():
    path = ".."
    pattern = "*.AIM"
    return AIMLoader(path, pattern)


class TestComposedDataset(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        ComposedDataset(get_file_loader(), PatchSampler())


if __name__ == '__main__':
    unittest.main()
