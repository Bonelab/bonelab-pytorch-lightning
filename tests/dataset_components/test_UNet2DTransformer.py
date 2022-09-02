import torch
import unittest

from blpytorchlightning.dataset_components.UNet2DTransformer import UNet2DTransformer


def get_torch_module():
    return torch.nn.Module()


class TestUNet2DTransformer(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        UNet2DTransformer(get_torch_module())


if __name__ == '__main__':
    unittest.main()
