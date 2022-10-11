from __future__ import annotations

import numpy as np

from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)


class Rescaler(BaseTransformer):
    """ A class to rescale the values in the input images from a given range to the range (-1,1). """

    def __init__(self, intensity_bounds: tuple[float, float]) -> None:
        """
        Initialization method.

        Parameters
        ----------
        intensity_bounds : tuple[float, float]
            The intensity bounds, in the same units of density as the image. Determine the linear mapping from the
            density space to the unitless [-1,1] interval for normalization.
        """
        self._intensity_bounds = intensity_bounds
        if not isinstance(self._intensity_bounds, tuple):
            raise ValueError("intensity bounds must be tuple of floats or ints of length 2")
        if not len(self._intensity_bounds) == 2:
            raise ValueError("intensity bounds must be tuple of floats or ints of length 2")
        for intensity in self._intensity_bounds:
            if not isinstance(intensity, float) or isinstance(intensity, int):
                raise ValueError("intensity bounds must be tuple of floats or ints of length 2")

    @property
    def intensity_bounds(self) -> tuple[float, float]:
        """
        `intensity_bounds` getter method.

        Returns
        -------
        tuple[float, float]
        """
        return self._intensity_bounds

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
