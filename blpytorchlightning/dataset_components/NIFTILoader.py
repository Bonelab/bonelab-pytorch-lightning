from __future__ import annotations

from bonelab.util.vtk_util import vtkImageData_to_numpy
from glob import glob
import numpy as np
import os
from vtk import vtkNIFTIImageReader

from blpytorchlightning.dataset_components.base_classes.BaseFileLoader import (
    BaseFileLoader,
)


class NIFTILoader(BaseFileLoader):

    def __init__(self, path: str, pattern: str = "*.nii") -> None:
        """

        Parameters
        ----------
        path: str
            Path to where the data is.

        pattern
        """
        self._path = path
        self._pattern = pattern
        self._build_image_list()

        self.reader = vtkNIFTIImageReader()

    @property
    def path(self) -> str:
        """

        Returns
        -------
        str
            The path.
        """
        return self._path

    @property
    def pattern(self) -> str:
        return self._pattern

    def _build_image_list(self) -> None:
        """
        Build list of images in the folder that the `path` property points to
        using the `pattern` property and the `glob` function.
        """
        self._image_list = glob(os.path.join(self._path, self._pattern))

    def __len__(self) -> int:
        """

        Returns
        -------

        """
        return len(self._image_list)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the tuple of image and masks for the given image.

        Parameters
        ----------
        idx : int
            The index into the list of images in the dataset.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            First element: the image as a numpy array of shape (1,D,H,W)
            Second element: the masks as a numpy array of shape (C,D,H,W)
        """
        image_fn = self._image_list[idx]

        self.reader.SetFileName(image_fn)
        self.reader.Update()
        image = vtkImageData_to_numpy(self.reader.GetOutput())

        image = np.expand_dims(image, 0)

        # TODO: more stuff for rescaling?

        # TODO: read masks
        masks = None

        return image, masks
