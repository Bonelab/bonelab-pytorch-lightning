from __future__ import annotations

import numpy as np
import torch

from blpytorchlightning.dataset_components.base_classes.BaseTransformer import (
    BaseTransformer,
)


class HRpQCTTransformer(BaseTransformer):
    """
    A class to apply the typical transformations required for processing an HR-pQCT image for deep learning.
    1. Rescale the densities in the image from density space to a unitless normalized interval using a truncated
    linear mapping with configurable density bounds.
    2. Convert the rescaled image and masks to torch tensors.
    """

    def __init__(
        self, intensity_bounds: tuple[float, float] = (-400, 1400), ohe: bool = False
    ) -> None:
        """
        Initialization method

        Parameters
        ----------
        intensity_bounds : tuple[float, float]
            The intensity bounds, in the same units of density as the image. Determine the linear mapping from the
            density space to the unitless [-1,1] interval for normalization. Default: [-400, 1400]

        ohe : bool
            A flag to determine whether to One-Hot Encode the targets. One-hot encoded targets would have a separate
            binary mask for each output class, while non-one-hot encoded targets have a single field where the integer
            value denotes the class of each voxel. Different loss functions require different target formats.
            Default: False.

        """
        self._intensity_bounds = intensity_bounds
        self._ohe = ohe

    @property
    def intensity_bounds(self) -> tuple[float, float]:
        """
        `intensity_bounds` getter method.

        Returns
        -------
        tuple[float, float]
        """
        return self._intensity_bounds

    @property
    def ohe(self) -> bool:
        """
        `ohe` getter method. This is the boolean flag that sets whether targets are one-hot encoded.

        Returns
        -------
        bool
        """
        return self._ohe

    def __call__(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A magic method that allows this function to be called as a function. Pass the sample through, rescale
        the image and convert everything to tensors.

        Parameters
        ----------
        sample: tuple[np.ndarray, np.ndarray]
            The input sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The transformed sample.
        """
        image, masks = sample
        image = self._rescale_image(image)
        image, masks = self._image_and_mask_to_tensors(image, masks)
        return image, masks

    def _rescale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rescale the image from density to unitless interval

        Parameters
        ----------
        image : np.ndarray
            The image in the density space.

        Returns
        -------
        np.ndarray
            The image rescaled to a unitless interval.
        """
        min_intensity = self._intensity_bounds[0]
        max_intensity = self._intensity_bounds[1]
        image = np.minimum(np.maximum(image, min_intensity), max_intensity)
        image = (2 * image - max_intensity - min_intensity) / (
            max_intensity - min_intensity
        )
        return image

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
        return image, masks
