import unittest

from blpytorchlightning.dataset_components.NIFTILoader import NIFTILoader


class TestNIFTILoader(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        path = "."
        pattern = "*.nii"
        NIFTILoader(path, pattern)


if __name__ == '__main__':
    unittest.main()