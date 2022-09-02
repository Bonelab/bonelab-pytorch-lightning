"""
Written by Nathan Neeteson
A module containing code for a synthesis/fusion network.
Synthesis3D is a class defining a 3D FCN that takes 3 preliminary predicted
segmentations as input and outputs a single final predicted segmentation.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Layer3D(nn.Module):
    """
    Layer3D - the repeating unit of the Synthesis model.

    See here for justification for layer ordering:
    https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
    """

    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel_size: int,
        padding: int,
        stride: int,
        groups: int,
        dropout: float,
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
            nn.Conv3d(
                inputs, outputs, kernel_size=kernel_size, padding=padding, stride=stride
            ),
            nn.ReLU(inplace=True),
            nn.GroupNorm(groups, outputs),
            nn.Dropout3d(dropout),
            nn.Conv3d(
                outputs,
                outputs,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.ReLU(inplace=True),
            nn.GroupNorm(groups, outputs),
            nn.Dropout3d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Should have shape (B,`inputs`,D,H,W).

        Returns
        -------
        torch.Tensor
            Will have shape (B,`outputs`,D,H,W)
        """
        return self.layer(x)


class Synthesis3D(nn.Module):
    """
    Synthesis3D - Takes several stacked 2D segmentations as input, returns a final 3D segmentation as output.
    """

    def __init__(
        self,
        input_channels: int,
        output_classes: int,
        num_filters: list[int],
        channels_per_group: int,
        dropout: float,
    ) -> None:
        """
        The initialization function

        Parameters
        ----------
        input_channels : int
            The number of channels in the expected input image.

        output_classes : int
            The number of classes (or embedding fields) you want the UNet to predict.

        num_filters : list[int]
            A list of integers, where the length of the list determines how many layers will be in the UNet, and the
            integer value of each element of the list determines how many filters will be in the `Layer3D` object
            at that level of the UNet. The first element corresponds to the layers closest to the inputs and outputs,
            while the last element corresponds to the deepest layer of the model.

        channels_per_group : int
            The number of channels/features that should be in each group when applying GroupNorm. Used to determine
            how many groups to split channels into in each layer.

        dropout : float
            The dropout rate for all layers in the model. Must be between 0 and 1, inclusive.

        """
        super().__init__()

        self.layer_kernel_size = 3
        self.layer_padding = (self.layer_kernel_size - 1) // 2  # for same conv
        self.layer_stride = 1

        self.layers = nn.ModuleList()
        f_prev = input_channels
        for f in num_filters:
            self.layers.append(
                Layer3D(
                    f_prev,
                    f,
                    self.layer_kernel_size,
                    self.layer_padding,
                    self.layer_stride,
                    num_filters[0] // channels_per_group,
                    dropout,
                )
            )
            f_prev = f

        if len(self.layers) > 0:
            self.map_to_output = nn.Conv3d(
                num_filters[-1], output_classes, kernel_size=1, stride=1
            )
        else:
            self.map_to_output = nn.Conv3d(
                self.num_inputs, output_classes, kernel_size=1, stride=1
            )

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.map_to_output(x)
