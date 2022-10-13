import unittest

from blpytorchlightning.dataset_components.samplers.ComposedSampler import ComposedSampler


class TestComposedSampler(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        ComposedSampler([])
