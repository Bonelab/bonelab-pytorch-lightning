from __future__ import annotations

import numpy as np

from blpytorchlightning.dataset_components.samplers.BaseSampler import BaseSampler
from blpytorchlightning.dataset_components.samplers.PatchSampler import PatchSampler


class ForegroundPatchSampler(PatchSampler):
    """A class to sample 2D or 3D patches from a 2D or 3D medical image and masks, with the patches centered on a
    certain class (or classes) in the masks that has/have been designated as the foreground."""

    def __init__(self,
                 patch_width: Union[int, List[int]] = 128,
                 foreground_channel: int = 0,
                 prob: float = 1.0
                 ) -> None:
        """
        Initialization method

        Parameters
        ----------
        patch_width : int
            The width of the patch to extract from the image and masks. Depending on the architecture of the GPU/CPU/TPU
            that you plan to train on, should probably be a power of 2. Default is `128`.

        foreground_channel : int
            The output class to center the patches on. Default is `0`

        prob : float
            The probability of selecting a foreground patch vs just selecting a random patch.
            Must be between 0.0 and 1.0, inclusive. Default is 1.0

        """
        super().__init__(patch_width)
        self._foreground_channel = foreground_channel
        self._prob = prob

        if not(isinstance(foreground_channel, int)) or (foreground_channel < 0):
            raise ValueError(f"`foreground_channel` must be a positive integer, got {foreground_channel}")

        if not(isinstance(prob, float)) or (prob < 0) or (prob > 1):
            raise ValueError(f"`prob` must be a float where 0.0 <= prob <= 1.0, got {prob}")

    @property
    def foreground_channel(self) -> int:
        """
        Getter method for `foreground_channel`

        Returns
        -------
        int
        """
        return self._foreground_channel

    @property
    def prob(self) -> float:
        """
        Getter method for `prob`

        Returns
        -------
        float
        """
        return self._prob

    def __call__(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pass the sample through this, extracting a volume patch

        Parameters
        ----------
        sample: tuple[np.ndarray, np.ndarray]
            The full input sample.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The patch sample.
        """
        if np.random.uniform(0, 1) < self._prob:
            return self._crop_to_foreground(*sample)
        else:
            return self._crop(*sample)

    def _crop_to_foreground(
        self, image: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop a random patch from the image and mask slices, centered on a foreground voxel. Compatible with 2D and 3D
        data, the restriction is that the spatial dimensions should be the last dimensions (batch, channels first).

        Parameters
        ----------
        image : np.ndarray
            The full image.

        masks : np.ndarray
            The full masks.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The patch sample.
        """
        image_shape = np.asarray(image[0, ...].shape)
        foreground_voxels = np.argwhere(
            masks[self._foreground_channel, ...] == 1
        ).transpose(0, 1)
        if len(foreground_voxels) > 0:
            patch_center = foreground_voxels[np.random.randint(len(foreground_voxels))]
        else:
            patch_center = image_shape // 2
        patch_start = patch_center - self.patch_width // 2
        patch_start = np.maximum(patch_start, 0)
        patch_start = patch_start - np.maximum(
            patch_start + self.patch_width - image_shape, 0
        )
        slicing_list = [slice(None)]
        if isinstance(self.patch_width, int):
            for ps in patch_start:
                slicing_list.append(slice(ps, ps + self._patch_width))
        else:
            for ps, pw in zip(patch_start, self.patch_width):
                slicing_list.append(slice(ps, ps + pw))
        return image[tuple(slicing_list)], masks[tuple(slicing_list)]
