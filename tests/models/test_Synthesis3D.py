import unittest

from blpytorchlightning.models.Synthesis3D import Layer3D, Synthesis3D


class TestLayer3D(unittest.TestCase):
    """ Test that we can import the module and work with the file """

    def test_can_instantiate(self):
        layer_spec = {
            "inputs": 8,
            "outputs": 16,
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
            "groups": 4,
            "dropout": 0.3
        }
        Layer3D(**layer_spec)


class TestSynthesis3D(unittest.TestCase):
    """ Test that we can import the module and work with the file """

    def test_can_instantiate(self):
        synthesis_spec = {
            "input_channels": 1,
            "output_classes": 3,
            "num_filters": [16, 32],
            "channels_per_group": 4,
            "dropout": 0.3
        }
        Synthesis3D(**synthesis_spec)
