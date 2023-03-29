from __future__ import annotations

import numpy as np
import torch

from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)


class TensorConverter(BaseTransformer):
    """A class to convert samples from numpy arrays to pytorch tensors"""

    def __init__(self, ohe: bool = False) -> None:
        """
        Initialization method.

        Parameters
        ----------
        ohe : bool
            A flag to determine whether to One-Hot Encode the targets. One-hot encoded targets would have a separate
            binary mask for each output class, while non-one-hot encoded targets have a single field where the integer
            value denotes the class of each voxel. Different loss functions require different target formats.
            Default: False.
        """
        self._ohe = ohe
        if not isinstance(self._ohe, bool):
            raise ValueError("ohe must be a bool")

    @property
    def ohe(self) -> bool:
        """
        `ohe` getter method.

        Returns
        -------
        bool
        """
        return self._ohe

    def __call__(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, masks = sample
        image, masks = self._image_and_mask_to_tensors(image, masks)
        return image, masks

    def _image_and_mask_to_tensors(
        self, image: np.ndarray, masks: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the image and mask to torch tensors.

        Parameters
        ----------
        image : np.ndarray
            The image.

        masks : np.ndarray
            The segmentation / masks.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            First element is the input image, second element is the segmentation / masks.
        """
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = torch.from_numpy(image).float()
        masks = torch.from_numpy(masks > 0).float()
        if masks.shape[0] > 1 and not self._ohe:
            masks = torch.argmax(masks, dim=0)
        elif masks.shape[0] == 1 and not self.ohe:
            # if there is only a single channel in the mask and we do not want a OHE then squeeze channels dim
            masks = masks.squeeze(0)
        return image, masks
