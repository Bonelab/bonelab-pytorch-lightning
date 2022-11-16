from __future__ import annotations

import torch

from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)


class TensorOneHotEncoder(BaseTransformer):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(
            self, sample: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        image, masks = sample
        masks = torch.movedim(torch.nn.functional.one_hot(masks, num_classes=self.num_classes), -1, 0)
        return image, masks


