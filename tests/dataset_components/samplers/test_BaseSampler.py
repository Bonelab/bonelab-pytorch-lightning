import unittest

from blpytorchlightning.dataset_components.samplers.BaseSampler import BaseSampler


class TestBaseSampler(unittest.TestCase):
    """ Test that we can import the module but cannot instantiate the abstract base class. """
    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            BaseSampler()


if __name__ == '__main__':
    unittest.main()
