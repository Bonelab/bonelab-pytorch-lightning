from __future__ import annotations

import torch

from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)


class TensorOneHotEncoder(BaseTransformer):
    def __init__(self, num_classes: int) -> None:
        """
        Initialization method.

        Parameters
        ----------
        num_classes : int
            How many classes will be in the targets.
        """
        self.num_classes = num_classes

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
        targets = torch.movedim(
            torch.nn.functional.one_hot(targets, num_classes=self.num_classes), -1, 0
        )
        return image, targets
