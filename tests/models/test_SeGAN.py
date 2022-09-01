import unittest
import torch

from blpytorchlightning.models.SeGAN import (
    EncoderLayer, DecoderLayer,
    Segmentor, Discriminator,
    get_segmentor_and_discriminators
)


class TestEncoderLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.layer_spec = {
            "inputs": 16,
            "outputs": 32,
            "kernel_size": 4,
            "padding": 1,
            "stride": 2,
            "groups": 4
        }
        self.encoder_layer = EncoderLayer(**self.layer_spec)

    def test_output(self):
        batch_size = 1
        height, width = 64, 64
        x = torch.zeros((batch_size, self.layer_spec["inputs"], height, width))
        y = self.encoder_layer(x)
        assert(isinstance(y, torch.Tensor))
        y_shape = list(y.shape)
        assert(y_shape[0] == batch_size)
        assert(y_shape[1] == self.layer_spec["outputs"])
        assert(y_shape[2] == height // 2)
        assert(y_shape[3] == width // 2)


class TestDecoderLayer(unittest.TestCase):

    def setUp(self):
        self.layer_spec = {
            "inputs": 128,
            "outputs": 32,
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
            "groups": 4,
            "scale_factor": 2,
            "scale_mode": "bilinear"
        }
        self.decoder_layer = DecoderLayer(**self.layer_spec)

    def test_output(self):
        batch_size = 1
        height, width = 64, 64
        x = torch.zeros((batch_size, self.layer_spec["inputs"], height, width))
        y = self.decoder_layer(x)
        assert (isinstance(y, torch.Tensor))
        y_shape = list(y.shape)
        assert (y_shape[0] == batch_size)
        assert (y_shape[1] == self.layer_spec["outputs"])
        assert (y_shape[2] == height * self.layer_spec["scale_factor"])
        assert (y_shape[3] == width * self.layer_spec["scale_factor"])


class TestSegmentor(unittest.TestCase):

    def setUp(self):
        self.segmentor_spec = {
            "input_channels": 1,
            "output_classes": 3,
            "num_filters": [64, 128, 256, 512],
            "channels_per_group": 16,
            "upsample_mode": "bilinear"
        }
        self.segmentor = Segmentor(**self.segmentor_spec)

    def test_output(self):
        batch_size = 1
        height, width = 128, 128
        x = torch.zeros((batch_size, self.segmentor_spec["input_channels"], height, width))
        y = self.segmentor(x)
        assert (isinstance(y, torch.Tensor))
        y_shape = list(y.shape)
        assert (y_shape[0] == batch_size)
        assert (y_shape[1] == self.segmentor_spec["output_classes"])
        assert (y_shape[2] == height)
        assert (y_shape[3] == width)


class TestDiscriminator(unittest.TestCase):

    def setUp(self):
        self.discriminator_spec = {
            "input_channels": 1,
            "num_filters": [64, 128, 256, 512],
            "channels_per_group": 16
        }
        self.discriminator = Discriminator(**self.discriminator_spec)

    def test_output(self):
        batch_size = 1
        height, width = 128, 128
        x = torch.zeros((batch_size, self.discriminator_spec["input_channels"], height, width))
        y = self.discriminator(x)
        assert (isinstance(y, list))
        assert(len(y) == len(self.discriminator_spec["num_filters"]))
        for z, nf in zip(y, self.discriminator_spec["num_filters"]):
            assert(isinstance(z, torch.Tensor))
            z_shape = list(z.shape)
            assert(z_shape[1] == nf)


class TestGetSegmentorAndDiscriminators(unittest.TestCase):

    def setUp(self):
        self.gan_spec = {
            "input_channels": 1,
            "output_classes": 3,
            "num_filters": [64, 128, 256, 512],
            "channels_per_group": 16,
            "upsample_mode": "bilinear"
        }

    def test_output(self):
        segmentor, discriminators = get_segmentor_and_discriminators(**self.gan_spec)
        assert(isinstance(segmentor, Segmentor))
        assert(isinstance(discriminators, list))
        for d in discriminators:
            assert(isinstance(d, Discriminator))

    def test_3d_output(self):
        segmentor, discriminators = get_segmentor_and_discriminators(**self.gan_spec, is_3d=True)
        assert (isinstance(segmentor, Segmentor))
        assert (isinstance(discriminators, list))
        for d in discriminators:
            assert (isinstance(d, Discriminator))

