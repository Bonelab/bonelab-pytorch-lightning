import unittest

from blpytorchlightning.dataset_components.base_classes.BaseTransformer import BaseTransformer


class TestTransformer(unittest.TestCase):
    """ Test that we can import the module but cannot instantiate the abstract base class. """
    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            BaseTransformer()


if __name__ == '__main__':
    unittest.main()