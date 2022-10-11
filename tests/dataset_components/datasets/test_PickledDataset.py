import unittest

from blpytorchlightning.dataset_components.datasets.PickledDataset import PickledData, PickledDataset


class TestPickledData(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        path = ".."
        PickledData(path)


class TestPickledDataset(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        folder = "."
        PickledDataset(folder)


if __name__ == '__main__':
    unittest.main()
