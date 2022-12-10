from __future__ import annotations

import numpy as np
import torch

from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)


class TensorRotater(BaseTransformer):
    """
    Transformer class for rotating an image and optionally also the targets. You would want to
    rotate the targets if you are doing segmentation but not if you are doing classification.
    """

    def __init__(self, dim: int = 3, rotate_targets: bool = True) -> None:
        """
        Initialization method.

        Parameters
        ----------
        dim : int
            The dimension of the images to be rotated.

        rotate_targets : bool
            Boolean flag to control whether to rotate the targets along with the image.
        """
        if not isinstance(rotate_targets, bool):
            raise ValueError("`rotate_targets` must be boolean")
        if not isinstance(dim, int):
            raise ValueError("`dim` must be int")
        self.rotate_targets = rotate_targets
        self.dim = dim

    def __call__(
        self, sample: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A magic method that allows this function to be called as a function. Pass the sample through, rescale
        the image and convert everything to tensors.

        Parameters
        ----------
        sample: tuple[torch.Tensor, torch.Tensor]
            The input sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The transformed sample.
        """
        image, targets = sample

        dims = np.random.randint(
            self.dim, size=(2,)
        )  # get two random integers for rotation dims
        dims = list(dims + 1)  # add 1 to skip over channel dim, and convert to a list

        # if the dims came up the same, this means do not rotate
        if dims[0] != dims[1]:
            image = torch.rot90(image, dims=dims)
            if self.rotate_targets:
                targets = torch.rot90(targets, dims=dims)

        return image, targets
