import unittest

from blpytorchlightning.dataset_components.AIMLoader import AIMLoader


class TestAIMLoader(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        path = "."
        pattern = "*.AIM"
        AIMLoader(path, pattern)


if __name__ == '__main__':
    unittest.main()
