import unittest

from blpytorchlightning.dataset_components.transformers.ComposedTransformers import ComposedTransformers


class TestComposedTransformers(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        ComposedTransformers([])


if __name__ == '__main__':
    unittest.main()
