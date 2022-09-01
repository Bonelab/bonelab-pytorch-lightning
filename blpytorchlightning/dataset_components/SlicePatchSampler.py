from __future__ import annotations

import numpy as np

from blpytorchlightning.dataset_components.PatchSampler import PatchSampler


class SlicePatchSampler(PatchSampler):
    """
    Class to sample a 2D slice from a 3D image and masks.
    """
    def __init__(self, patch_width=128, dims: Union[Set[int], List[int], Tuple[int]] = (0, 1, 2), foreground_channel=0):
        super().__init__(patch_width, foreground_channel)
        try:
            self._dims = set(map(lambda x: int(x), dims))
        except ValueError as e:
            raise ValueError(
                f"`dim` arg accepts only iterables containing values that are ints or can be cast to ints...\n{e}`"
            )
        if max(self._dims) > 2 or min(self._dims) < 0:
            raise ValueError("`dims` must contain only values between 0 and 2, inclusive")

    @property
    def dims(self) -> Set[int]:
        """
        Getter method for the `dims` property.

        Returns
        -------
        Set[int]
        """
        return self._dims

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pass the sample through this, extracting a slice patch

        Parameters
        ----------
        sample: Tuple[np.ndarray, np.ndarray]
            The full 3D input sample.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The 2D slice patch sample.
        """
        sample = self._get_random_slice(*sample)
        return self._crop_to_foreground(*sample)

    def _get_random_slice(self, image: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random slice in a random dim from the image and mask

         Parameters
        ----------
        image : np.ndarray
            The full image.

        masks : np.ndarray
            The full masks.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The slice sample.
        """
        slicing_dim = np.random.choice(list(self._dims)) + 1
        slicing_list = [slice(None)] * 4
        slice_index = np.random.randint(image.shape[slicing_dim])
        slicing_list[slicing_dim] = slice_index
        return image[tuple(slicing_list)], masks[tuple(slicing_list)]
