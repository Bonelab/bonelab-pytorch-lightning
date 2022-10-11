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
        try:
            self._dims = tuple(map(lambda x: float(x), intensity_bounds))
        except ValueError as e:
            raise ValueError(
                f"`intensity_bounds` arg accepts only iterables containing values that are floats or can be cast to "
                f"floats...\n{e}"
            )
        if len(self._dims) != 2:
            raise ValueError("`intensity_bounds` arg must be a length-2 iterable")

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
