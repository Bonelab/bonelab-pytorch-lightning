from __future__ import annotations

import numpy as np
import os
import vtkbone
from glob import glob
from bonelab.util.aim_calibration_header import get_aim_density_equation
from bonelab.util.vtk_util import vtkImageData_to_numpy
from enum import Enum

from blpytorchlightning.dataset_components.base_classes.BaseFileLoader import BaseFileLoader

# create an ImageType Enum with two options: density image or mask
ImageType = Enum("ImageType", "DENSITY MASK")


class AIMLoader(BaseFileLoader):
    """ Class for loading AIM images (and masks) from a directory, aligning them, and returning as numpy arrays."""

    def __init__(self, path: str, pattern: str) -> None:
        """ Initialization method.

        Parameters
        ----------
        path : str
            The path of the directory containing the data to load.

        pattern : str
            The pattern to use to `glob` to find the data in that directory.
        """
        self._path = path
        self._pattern = pattern
        self._build_image_list()

        self.reader = vtkbone.vtkboneAIMReader()
        self.reader.DataOnCellsOff()

    def _build_image_list(self) -> None:
        """
        Build list of images in the folder that the `path` property points to
        using the `pattern` property and the `glob` function.
        """
        self._image_list = glob(os.path.join(self._path, self._pattern))

    @property
    def path(self) -> str:
        """
        Output the path to the folder of images.

        Returns
        -------
        str
            The `path` property.
        """
        return self._path

    @property
    def pattern(self) -> str:
        """
        Output the pattern being used to find images.

        Returns
        -------
        str
            The `pattern` property.
        """
        return self._pattern

    def __len__(self) -> int:
        """
        Output number of images in the dataset.

        Returns
        -------
        int
            The number of images in the path that fit the pattern.
        """
        return len(self._image_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the tuple of image and masks for the given image.

        Parameters
        ----------
        idx : int
            The index into the list of images in the dataset.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            First element: the density image as a numpy array of shape (1,D,H,W)
            Second element: the masks as a numpy array of shape (3,D,H,W).
            Mask order is cortical, trabecular, background
        """
        image_fn = self._image_list[idx]
        return self._get_image_and_masks(image_fn)

    def _get_image_and_masks(self, image_fn: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image and mask for a given image name.

        Parameters
        ----------
        image_fn : str
            The full filepath to the image on disk.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            First element: the density image as a numpy array of shape (1,D,H,W)
            Second element: the masks as a numpy array of shape (3,D,H,W).
            Mask order is cortical, trabecular, background
        """
        cort_mask_fn = image_fn.replace('.AIM', '_CORT_MASK.AIM')
        trab_mask_fn = image_fn.replace('.AIM', '_TRAB_MASK.AIM')
        image, image_pos = self._get_image_data_and_position(image_fn, ImageType.DENSITY)
        cort_mask, cort_mask_pos = self._get_image_data_and_position(cort_mask_fn, ImageType.MASK)
        trab_mask, trab_mask_pos = self._get_image_data_and_position(trab_mask_fn, ImageType.MASK)
        image, cort_mask, trab_mask = tuple(self._align_aims([
            (image, image_pos, "edge"),
            (cort_mask, cort_mask_pos, "constant"),
            (trab_mask, trab_mask_pos, "constant")
        ]))
        back_mask = np.logical_and(np.logical_not(cort_mask), np.logical_not(trab_mask))
        image = np.expand_dims(image, 0)
        masks = np.stack([cort_mask, trab_mask, back_mask])
        return image, masks

    def _get_image_data_and_position(self,
                                     fn: str,
                                     image_type: ImageType
                                     ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract the image data and position vector from an AIM file.

        Parameters
        ----------
        fn : str
            The full filepath to the image on disk.

        image_type : ImageType
            The type of image being loaded - DENSITY or MASK. Determines if image values are transformed to densities
            using calibration information or if they are transformed to boolean values.

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            First element: the image as a numpy array
            Second element: a length 3 list of integers that describes the positional offset (in voxels) of the image
            relative to a shared origin.

        """
        self.reader.SetFileName(fn)
        self.reader.Update()
        data = vtkImageData_to_numpy(self.reader.GetOutput())
        if image_type is ImageType.DENSITY:
            m, b = get_aim_density_equation(self.reader.GetProcessingLog())
            data = (m * data + b).astype(float)
        elif image_type is ImageType.MASK:
            data = (data > 0).astype(int)
        position = list(self.reader.GetPosition())
        return data, position

    @staticmethod
    def _align_aims(data: List[Tuple[np.ndarray, List[int], str]]) -> List[np.ndarray]:
        """
        A static method that takes a set of images/masks and their positions and aligns them.

        Parameters
        ----------
        data : List[Tuple[np.ndarray, List[int], str]]
            A list of tuples, where each tuple contains an image/mask numpy array, a length-3 list of integers
            describing its positional offset (in voxels) from a shared origin, and the pad mode ('constant' or 'edge')

        Returns
        -------
        List[np.ndarray]
            A list of aligned images, with the same shape, in the same order as given to the `data` arg

        """
        min_position = np.asarray([p for _, p, _ in data]).min(axis=0)
        pad_lower = [p - min_position for _, p, _ in data]
        max_shape = np.asarray([(aim.shape + pl) for (aim, _, _), pl in zip(data, pad_lower)]).max(axis=0)
        pad_upper = [(max_shape - (aim.shape + pl)) for (aim, _, _), pl in zip(data, pad_lower)]
        return [
            np.pad(aim, tuple([(l, u) for l, u in zip(pl, pu)]), m)
            for (aim, _, m), pl, pu in zip(data, pad_lower, pad_upper)
        ]