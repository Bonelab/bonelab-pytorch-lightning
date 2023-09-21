from __future__ import annotations

import numpy as np

from blpytorchlightning.dataset_components.samplers.BaseSampler import BaseSampler


class PatchSampler(BaseSampler):
    """A class to sample 2D or 3D patches from a 2D or 3D medical image and masks."""

    def __init__(self, patch_width: Union[int, List[int]] = 128) -> None:
        """
        Initialization method

        Parameters
        ----------
        patch_width : int
            The width of the patch to extract from the image and masks. Depending on the architecture of the GPU/CPU/TPU
            that you plan to train on, should probably be a power of 2. Default is `128`.
        """
        if isinstance(patch_width, int):
            if patch_width < 0:
                raise ValueError(f"`patch_width` must be a positive integer, got {patch_width}")
            self._patch_width = patch_width
        if isinstance(patch_width, list):
            if any([pw < 0 for pw in patch_width]):
                raise ValueError(f"`patch_width` must be a list of positive integers, got {patch_width}")
            self._patch_width = np.array(patch_width, dtype=int)
        else:
            raise ValueError(f"`patch_width` must be an integer or a list of integers, got {patch_width}")

    @property
    def patch_width(self) -> Union[int, np.ndarray]:
        """
        Getter method for `patch_width`

        Returns
        -------
        int
        """
        return self._patch_width

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
        return self._crop(*sample)

    def _crop(
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
        patch_center = np.asarray([np.random.randint(img_sh) for img_sh in image_shape])
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
