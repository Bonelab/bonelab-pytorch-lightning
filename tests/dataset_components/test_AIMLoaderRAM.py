import unittest

from blpytorchlightning.dataset_components.AIMLoaderRAM import AIMLoaderRAM


class TestAIMLoaderRAM(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        path = "."
        pattern = "*.AIM"
        AIMLoaderRAM(path, pattern)


if __name__ == '__main__':
    unittest.main()