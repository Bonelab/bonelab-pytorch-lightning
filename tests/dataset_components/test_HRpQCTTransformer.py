import unittest

from blpytorchlightning.dataset_components.HRpQCTTransformer import HRpQCTTransformer


class TestHRpQCTTransformer(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        HRpQCTTransformer()


if __name__ == '__main__':
    unittest.main()