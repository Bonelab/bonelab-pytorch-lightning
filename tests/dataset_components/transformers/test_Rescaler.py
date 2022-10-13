import unittest

from blpytorchlightning.dataset_components.transformers.Rescaler import Rescaler


class TestRescaler(unittest.TestCase):
    """ Test that we can import and work with the module. """

    def test_can_instantiate(self):
        Rescaler([-100, 100])
