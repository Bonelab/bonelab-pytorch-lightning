import unittest

from blpytorchlightning.dataset_components.transformers.TensorConverter import TensorConverter


class TestTensorConverter(unittest.TestCase):
    """ Test that we can import and work with the module. """

    def test_can_instantiate(self):
        TensorConverter()
