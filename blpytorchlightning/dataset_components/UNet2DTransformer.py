from __future__ import annotations

import torch

from blpytorchlightning.dataset_components.base_classes.BaseTransformer import (
    BaseTransformer,
)


class UNet2DTransformer(BaseTransformer):
    """
    A class for using a pre-trained 2D UNet to create predicted segmentations by sweeping over the input image slice
    by slice. The input image is (1xHxWxD) and the output image is (4xHxWxD). The first channel is the image, while the
    subsequent channels are the predictions generated by sweeping over the image along the spatial dimensions in the
    order of the dimensions of the image (i.e. H then W then D, whatever those happen to be).
    """

    def __init__(self, model2d: torch.nn.Module) -> None:
        """
        Initialization method

        Parameters
        ----------
        model2d : torch.nn.Module
            The 2D segmentation model that will be used to compute the predicted slice-wise segmentations.

        """
        self._model2d = model2d
        self._model2d.eval()

    @property
    def model2d(self) -> torch.nn.Module:
        """
        Getter method for the `model2d` property.

        Returns
        -------
        torch.nn.Module
        """
        return self._model2d

    def __call__(
        self, sample: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Use the 2D model to convert the image from a (1xHxWxD) tensor to
        a (4xHxWxD) tensor where the channels are the image and axial, coronal,
        and sagittal slice-by-slice predictions. Inputs must be tensors since they will be fed directly to a model.

        Parameters
        ----------
        sample: tuple[torch.Tensor, torch.Tensor]
            The sample, with a 3D image and masks.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The transformed sample - the masks are unchanged, but the image has 3 extra channels, corresponding to
            the predicted segmentations generated using the `model2d` segmentation model to sweep slice-wise over
            the image and generate a predicted segmentation for each direction it could sweep over the image.

        """
        image, mask = sample
        image = self._construct_full_image(image)
        return image, mask

    def _construct_full_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Assemble the full input from the base image.

        Parameters
        ----------
        image : torch.Tensor
            The input image.

        Returns
        -------
        torch.Tensor
            The output image, with the predicted slice-wise predictions stacked on the channel dimension.
        """
        image_shape = list(image.shape)
        image_shape[0] = 4
        full_image = torch.zeros(image_shape)
        full_image[0, ...] = image
        for j in [1, 2, 3]:
            full_image[j, ...] = self._get_predictions(image, j)
        return full_image

    def _get_predictions(self, image: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Transpose an image so that the axis we want to sweep over resembles
        the batch axis and then feed through the 2D model.

        Parameters
        ----------
        image : torch.Tensor
            The input image.

        axis : int
            The dimension that will be transposed with the batch dimension so that a slice-wise prediction can be
            computed for the entire image at once.

        Returns
        -------
        torch.Tensor
            The slice-wise (along the specified dimension) predicted segmentation of the image.

        """
        self._model2d.eval()
        image = image.squeeze(0).transpose(0, axis - 1).unsqueeze(1)
        with torch.no_grad():
            return self._model2d(image).squeeze(1).transpose(0, axis - 1)
