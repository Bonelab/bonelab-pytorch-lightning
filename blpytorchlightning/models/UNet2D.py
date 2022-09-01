"""
Written by Nathan Neeteson
A module containing code for a modified UNet.
Layer2D is a class containing a repeating unit used in the UNet.
UNet2D is a class containing the full structure of the UNet, with a
configurable number of layers and number of filters in each layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# the repeating structure of the UNet, which occurs in each layer on the
# way down and up the encoder and decoder
class Layer2D(nn.Module):
    """
    Layer2D - the repeating unit of the 2D UNet.

    See here for justification for layer ordering:
    https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
    """

    def __init__(self,
                 inputs: int,
                 outputs: int,
                 kernel_size: int,
                 padding: int,
                 stride: int,
                 groups: int,
                 dropout: float
                 ) -> None:
        """
        Initialization method for the Layer2D class.

        Parameters
        ----------
        inputs : int
            Size of feature dimension in the expected input tensors.

        outputs : int
            Size of feature dimension in output tensors.

        kernel_size : int
            Kernel size for the two convolutional layers.

        padding : int
            Padding size for the two convolutional layers.

        stride : int
            Stride size for the two convolutional layers.

        groups : int
            Number of groups to separate features into in the two GroupNorm layers.

        dropout : float
            The dropout rate, between 0 and 1 inclusive, for the two Dropout layers.

        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.GroupNorm(groups, outputs),
            nn.Dropout2d(dropout),
            nn.Conv2d(outputs, outputs, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.GroupNorm(groups, outputs),
            nn.Dropout2d(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Should have shape (B,`inputs`,H,W).

        Returns
        -------
        torch.Tensor
            Will have shape (B,`outputs`,H,W)
        """
        return self.layer(x)


class UNet2D(nn.Module):
    """
    UNet2D - Takes an image as input, returns a segmentation (or level-set embedding of segmentation) as output.
    """

    def __init__(self,
                 input_channels: int,
                 output_classes: int,
                 num_filters: List[int],
                 channels_per_group: int,
                 dropout: float,
                 upsample_mode: str = "bilinear"
                 ) -> None:
        """
        The initialization function

        Parameters
        ----------
        input_channels : int
            The number of channels in the expected input image.

        output_classes : int
            The number of classes (or embedding fields) you want the UNet to predict.

        num_filters : List[int]
            A list of integers, where the length of the list determines how many layers will be in the UNet, and the
            integer value of each element of the list determines how many filters will be in the `Layer2D` object
            at that level of the UNet. The first element corresponds to the layers closest to the inputs and outputs,
            while the last element corresponds to the deepest layer of the UNet.

        channels_per_group : int
            The number of channels/features that should be in each group when applying GroupNorm. Used to determine
            how many groups to split channels into in each layer.

        dropout : float
            The dropout rate for all layers in the UNet. Must be between 0 and 1, inclusive.

        upsample_mode : str
            The mode of upsampling to use on the decoder half of the network. Can be one of 'nearest', 'linear',
            'bilinear', 'bicubic' and 'trilinear'. Default: 'bilinear'.

        """
        super().__init__()
        # layers
        self.channels_per_group = channels_per_group
        self.dropout = dropout
        self.layer_kernel_size = 3
        self.layer_padding = (self.layer_kernel_size - 1) // 2  # for same conv
        self.layer_stride = 1
        # down and upsampling
        self.scale_factor = 2
        self.upsample_mode = upsample_mode
        # initialize some module lists for the 4 types of operations
        self.layer_down = nn.ModuleList()
        self.down = nn.ModuleList()
        self.layer_up = nn.ModuleList()
        self.up = nn.ModuleList()
        self.layer_down.append(
            Layer2D(
                input_channels, num_filters[0],
                self.layer_kernel_size, self.layer_padding, self.layer_stride,
                num_filters[0] // self.channels_per_group, self.dropout
            )
        )
        for fi in range(1, len(num_filters)):
            self.down.append(nn.MaxPool2d(self.scale_factor))
            self.layer_down.append(
                Layer2D(
                    num_filters[fi - 1], num_filters[fi],
                    self.layer_kernel_size, self.layer_padding, self.layer_stride,
                    num_filters[fi] // self.channels_per_group, self.dropout
                )
            )
            self.up.append(
                nn.Upsample(
                    scale_factor=self.scale_factor,
                    mode=self.upsample_mode
                )
            )
            self.layer_up.append(
                Layer2D(
                    2 * num_filters[fi - 1], num_filters[fi - 1],
                    self.layer_kernel_size, self.layer_padding, self.layer_stride,
                    num_filters[fi - 1] // self.channels_per_group, self.dropout
                )
            )
        self.map_to_output = nn.Conv2d(
            num_filters[0], output_classes,
            kernel_size=1, stride=1
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
            Segmentation, or level-set embedding fields, predicted from image. Shape (B,`output_classes`,H,W)
        """
        x_down = [self.layer_down[0](x)]
        for layer_down, down in zip(self.layer_down[1:], self.down):
            x_down.append(layer_down(down(x_down[-1])))

        x = x_down.pop()
        for layer_up, up in zip(reversed(self.layer_up), reversed(self.up)):
            x = up(x)
            x = layer_up(torch.cat([x_down.pop(), x], dim=1))

        return self.map_to_output(x)
