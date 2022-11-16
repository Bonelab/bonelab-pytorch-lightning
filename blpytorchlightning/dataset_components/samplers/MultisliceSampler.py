from __future__ import annotations

import numpy as np
from collections.abc import Iterable

from blpytorchlightning.dataset_components.samplers.BaseSampler import BaseSampler


def wrap_element_by_reflection(x: int, min_val: int, max_val: int):
    """
    A function to wrap an element of an iterable by reflection.

    Parameters
    ----------
    x : int
    min_val : int
    max_val : int

    Returns
    -------
    int
    """
    if x < min_val:
        return min_val + (min_val - x)
    elif x >= max_val:
        return max_val - (x - max_val)
    else:
        return x


class MultisliceSampler(BaseSampler):
    """
    Class to sample a stack of 2D slices from a 3D image and masks. Only the center slice
    is cut from the mask. Only works on images with a single channel, since the multiple slices are
    going to be concatenated onto the channel dimension.
    If you want to use this with an image with multiple channels you will have to figure out how
    to make that work :)
    """

    def __init__(
        self, dims: Iterable[int] = (0, 1, 2), num_adjacent_slices: int = 2
    ) -> None:
        """
        Initialization method.

        Parameters
        ----------
        dims : Iterable[int]
            Dimensions along which to cut slices.

        num_adjacent_slices : int
            Number of adjacent slices to center slice to extract from image.
        """
        try:
            self._dims = set(map(lambda x: int(x), dims))
        except ValueError as e:
            raise ValueError(
                f"`dim` arg accepts only iterables containing values that are ints or can be cast to ints...\n{e}"
            )
        if max(self._dims) > 2 or min(self._dims) < 0:
            raise ValueError(
                "`dims` must contain only values between 0 and 2, inclusive"
            )
        self._num_adjacent_slices = num_adjacent_slices
        if not isinstance(self._num_adjacent_slices, int):
            raise ValueError("`num_adjacent_slices` must be an integer")

    @property
    def dims(self) -> set[int]:
        """
        Getter method for the `dims` property.

        Returns
        -------
        Set[int]
        """
        return self._dims

    @property
    def num_adjacent_slices(self) -> int:
        """

        Returns
        -------
        int
            Number of adjacent slices to cut from image.
        """
        return self._num_adjacent_slices

    def __call__(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pass the sample through this, extracting a slice patch

        Parameters
        ----------
        sample: tuple[np.ndarray, np.ndarray]
            The full 3D input sample.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The 2D slice patch sample.
        """
        return self._get_random_multislice(*sample)

    def _get_random_multislice(
        self, image: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a random stack of slices in a random dim from the image and mask

         Parameters
        ----------
        image : np.ndarray
            The full image.

        masks : np.ndarray
            The full masks.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The slice sample.
        """
        assert image.shape[0] == 1
        slicing_dim = np.random.choice(list(self._dims)) + 1
        slicing_list_stack = [slice(None)] * 4
        slicing_list_center = [slice(None)] * 4
        center_slice_index = np.random.randint(image.shape[slicing_dim])
        stack_slice_indices = [center_slice_index]
        for n in range(self._num_adjacent_slices):
            stack_slice_indices.insert(0, center_slice_index - n - 1)
            stack_slice_indices.append(center_slice_index + n + 1)
        stack_slice_indices = list(
            map(
                lambda x: wrap_element_by_reflection(
                    x, 0, image.shape[slicing_dim] - 1
                ),
                stack_slice_indices,
            )
        )
        slicing_list_stack[slicing_dim] = stack_slice_indices
        slicing_list_center[slicing_dim] = center_slice_index
        image = np.moveaxis(image[tuple(slicing_list_stack)], slicing_dim, 1).squeeze(0)
        masks = masks[tuple(slicing_list_center)]
        return image, masks
