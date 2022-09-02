import unittest

from blpytorchlightning.models.UNet import Layer, UNet


class TestLayer2D(unittest.TestCase):
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
        Layer(**layer_spec)


class TestUNet2D(unittest.TestCase):
    """ Test that we can import the module and work with the file """

    def test_can_instantiate(self):
        unet_spec = {
            "input_channels": 1,
            "output_classes": 3,
            "num_filters": [16, 32],
            "channels_per_group": 4,
            "dropout": 0.3
        }
        UNet(**unet_spec)
