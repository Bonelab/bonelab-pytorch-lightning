"""
Written by Nathan Neeteson.
A module containing code for a modified SeGAN. Default is 2D, but can be swapped to 3D.
Architecture of models contained herein is based on, but not identical to, the SeGAN model
proposed in this paper: https://doi.org/10.1007/s12021-018-9377-x
EncoderLayer is a class containing the repeating unit used in the segmentor and discriminator
DecoderLayer is a class containing the repeating unit used in the segmentor
Segmentor is a class containing the segmentor model of the SeGAN
Discriminator is a class containing the discriminator model of the SeGAN
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    """
    The repeating unit of the encoder.
    """

    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel_size: int,
        padding: int,
        stride: int,
        groups: int,
        is_3d: bool = False,
    ) -> None:
        """
        Initialization method.

        Parameters
        ----------
        inputs : int
            Size of feature dimension in the expected input tensors.

        outputs : int
            Size of feature dimension in output tensors.

        kernel_size : int
            Kernel size for the convolutional layer.

        padding : int
            Padding size for the convolutional layer.

        stride : int
            Stride size for the convolutional layer.

        groups : int
            Number of groups to separate features into in the GroupNorm layer.

        is_3d : bool
            Flag that can be set to swap 2D layers for 3D layers. Default: `False`
        """
        super().__init__()

        conv = nn.Conv3d if is_3d else nn.Conv2d

        self.layer = nn.Sequential(
            conv(
                inputs, outputs, kernel_size=kernel_size, padding=padding, stride=stride
            ),
            nn.GroupNorm(groups, outputs),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
        """
        return self.layer(x)


class DecoderLayer(nn.Module):
    """
    The repeating unit of the decoder.
    """

    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel_size: int,
        padding: int,
        stride: int,
        groups: int,
        scale_factor: int,
        scale_mode: str = "bilinear",
        is_3d: bool = False,
    ) -> None:
        """
        Initialization method.

        Parameters
        ----------
        inputs : int
            Size of feature dimension in the expected input tensors.

        outputs : int
            Size of feature dimension in output tensors.

        kernel_size : int
            Kernel size for the convolutional layer.

        padding : int
            Padding size for the convolutional layer.

        stride : int
            Stride size for the convolutional layer.

        groups : int
            Number of groups to separate features into in the GroupNorm layer.

        scale_factor : int
            The factor by which to rescale the image by using Upsample.

        scale_mode : str
            The mode of upsampling to use on the decoder half of the network. Can be one of 'nearest', 'linear',
            'bilinear', 'bicubic' and 'trilinear'. Default: 'bilinear'.

        is_3d : bool
            Flag that can be set to swap 2D layers for 3D layers. Default: `False`
        """
        super().__init__()

        conv = nn.Conv3d if is_3d else nn.Conv2d

        self.layer = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode=scale_mode, align_corners=False
            ),
            conv(
                inputs, outputs, kernel_size=kernel_size, padding=padding, stride=stride
            ),
            nn.GroupNorm(groups, outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
        """
        return self.layer(x)


class Segmentor(nn.Module):
    """
    Generates a segmentation from an image.
    """

    def __init__(
        self,
        input_channels: int,
        output_classes: int,
        num_filters: list[int],
        channels_per_group: int,
        upsample_mode: str = "bilinear",
        is_3d: bool = False,
    ) -> None:
        """
        The initialization function

        Parameters
        ----------
        input_channels : int
            The number of channels in the expected input image.

        output_classes : int
            The number of classes (or embedding fields) you want the Segmentor to predict.

        num_filters : list[int]
            A list of integers, where the length of the list determines how many layers will be in the UNet, and the
            integer value of each element of the list determines how many filters will be in the `Layer2D` object
            at that level of the UNet. The first element corresponds to the layers closest to the inputs and outputs,
            while the last element corresponds to the deepest layer of the UNet.

        channels_per_group : int
            The number of channels/features that should be in each group when applying GroupNorm. Used to determine
            how many groups to split channels into in each layer.

        upsample_mode : str
            The mode of upsampling to use on the decoder half of the network. Can be one of 'nearest', 'linear',
            'bilinear', 'bicubic' and 'trilinear'. Default: 'bilinear'.

        is_3d : bool
            Flag that can be set to swap 2D layers for 3D layers. Default: `False`
        """
        super().__init__()
        # encoder params
        encoder_kernel_size = 4
        encoder_padding = 1
        encoder_stride = 2
        # decoder params
        decoder_kernel_size = 3
        decoder_padding = 1
        decoder_stride = 1
        decoder_resize_factor = 2
        self.encoder = nn.ModuleList()
        self.encoder.append(
            EncoderLayer(
                input_channels,
                num_filters[0],
                encoder_kernel_size,
                encoder_padding,
                encoder_stride,
                num_filters[0] // channels_per_group,
                is_3d,
            )
        )
        for fi in range(1, len(num_filters)):
            self.encoder.append(
                EncoderLayer(
                    num_filters[fi - 1],
                    num_filters[fi],
                    encoder_kernel_size,
                    encoder_padding,
                    encoder_stride,
                    num_filters[fi] // channels_per_group,
                    is_3d,
                )
            )
        self.decoder = nn.ModuleList()
        for fi in range(1, len(num_filters) - 1):
            self.decoder.append(
                DecoderLayer(
                    2 * num_filters[fi],
                    num_filters[fi - 1],
                    decoder_kernel_size,
                    decoder_padding,
                    decoder_stride,
                    num_filters[fi - 1] // channels_per_group,
                    decoder_resize_factor,
                    upsample_mode,
                    is_3d,
                )
            )
        self.decoder.append(
            DecoderLayer(
                num_filters[-1],
                num_filters[-2],
                decoder_kernel_size,
                decoder_padding,
                decoder_stride,
                num_filters[-2] // channels_per_group,
                decoder_resize_factor,
                upsample_mode,
                is_3d,
            )
        )

        conv = nn.Conv3d if is_3d else nn.Conv2d

        self.map_to_output = nn.Sequential(
            nn.Upsample(
                scale_factor=decoder_resize_factor,
                mode=upsample_mode,
                align_corners=False,
            ),
            conv(
                2 * num_filters[0],
                output_classes,
                kernel_size=decoder_kernel_size,
                padding=decoder_padding,
                stride=decoder_stride,
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method.

        Parameters
        ----------
        x : torch.Tensor
            The input image to process. Must have shape (B,`input_channels`,H,W)

        Returns
        -------
        torch.Tensor
            Segmentation predicted from image. Shape (B,`output_classes`,H,W)
        """
        x_down = [self.encoder[0](x)]
        # noinspection PyTypeChecker
        for encoder_layer in self.encoder[1:]:
            x_down.append(encoder_layer(x_down[-1]))
        x = self.decoder[-1](x_down.pop())
        # noinspection PyTypeChecker
        for decoder_layer in reversed(self.decoder[:-1]):
            x = decoder_layer(torch.cat([x, x_down.pop()], dim=1))
        return self.map_to_output(torch.cat([x, x_down.pop()], dim=1))


class Discriminator(nn.Module):
    """
    Attempts to create multi-scale feature maps that can be used to discriminate between ground-truth
    segmentations and those generated by the segmentor.
    """

    def __init__(
        self,
        input_channels: int,
        num_filters: list[int],
        channels_per_group: int,
        is_3d: bool = False,
    ) -> None:
        """
        The initialization function

        Parameters
        ----------
        input_channels : int
            The number of channels in the expected input image.

        num_filters : list[int]
            A list of integers, where the length of the list determines how many layers will be in the UNet, and the
            integer value of each element of the list determines how many filters will be in the `Layer2D` object
            at that level of the UNet. The first element corresponds to the layers closest to the inputs and outputs,
            while the last element corresponds to the deepest layer of the UNet.

        channels_per_group : int
            The number of channels/features that should be in each group when applying GroupNorm. Used to determine
            how many groups to split channels into in each layer.

        is_3d : bool
            Flag that can be set to swap 2D layers for 3D layers. Default: `False`
        """
        super().__init__()
        # encoder params
        encoder_kernel_size = 4
        encoder_padding = 1
        encoder_stride = 2
        self.encoder = nn.ModuleList()
        self.encoder.append(
            EncoderLayer(
                input_channels,
                num_filters[0],
                encoder_kernel_size,
                encoder_padding,
                encoder_stride,
                num_filters[0] // channels_per_group,
                is_3d,
            )
        )
        for fi in range(1, len(num_filters)):
            self.encoder.append(
                EncoderLayer(
                    num_filters[fi - 1],
                    num_filters[fi],
                    encoder_kernel_size,
                    encoder_padding,
                    encoder_stride,
                    num_filters[fi] // channels_per_group,
                    is_3d,
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass method.

        Parameters
        ----------
        x : torch.Tensor
            The input image, masked by a segmentation.

        Returns
        -------
        list[torch.Tensor]
            List of features maps, one for each scale in the encoder.
        """
        features = [self.encoder[0](x)]
        # noinspection PyTypeChecker
        for encoder_layer in self.encoder[1:]:
            features.append(encoder_layer(features[-1]))
        return features


def get_segmentor_and_discriminators(
    input_channels: int,
    output_classes: int,
    num_filters: list[int],
    channels_per_group: int,
    upsample_mode: str = "bilinear",
    is_3d: bool = False,
) -> tuple[Segmentor, list[Discriminator]]:
    """
    Convenience function to quickly get segmentor and discriminators for a S<n>-<n>C SeGAN, where
    n is the number of output classes.

    Parameters
    ----------
    input_channels : int
        The number of channels in the expected input image.

    output_classes : int
        The number of classes (or embedding fields) you want the Segmentor to predict.

    num_filters : list[int]
        A list of integers, where the length of the list determines how many layers will be in the UNet, and the
        integer value of each element of the list determines how many filters will be in the `Layer2D` object
        at that level of the UNet. The first element corresponds to the layers closest to the inputs and outputs,
        while the last element corresponds to the deepest layer of the UNet.

    channels_per_group : int
        The number of channels/features that should be in each group when applying GroupNorm. Used to determine
        how many groups to split channels into in each layer.

    upsample_mode : str
        The mode of upsampling to use on the decoder half of the network. Can be one of 'nearest', 'linear',
        'bilinear', 'bicubic' and 'trilinear'. Default: 'bilinear'.

    is_3d : bool
            Flag that can be set to swap 2D layers for 3D layers. Default: `False`

    Returns
    -------
    tuple[Segmentor, list[Discriminator]]
        The segmentor and discriminator models for the S<n>-<n>C SeGAN.
    """

    segmentor = Segmentor(
        input_channels,
        output_classes,
        num_filters,
        channels_per_group,
        upsample_mode,
        is_3d,
    )

    discriminators = [
        Discriminator(input_channels, num_filters, channels_per_group, is_3d)
        for _ in range(output_classes)
    ]

    return segmentor, discriminators
