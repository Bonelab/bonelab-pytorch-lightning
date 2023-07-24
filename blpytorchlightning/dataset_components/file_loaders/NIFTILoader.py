from __future__ import annotations

from bonelab.util.vtk_util import vtkImageData_to_numpy
from glob import glob
import numpy as np
import os
from vtk import vtkNIFTIImageReader

from blpytorchlightning.dataset_components.file_loaders.BaseFileLoader import (
    BaseFileLoader,
)


class NIFTILoader(BaseFileLoader):
    def __init__(self,
                 path: str,
                 labels: list[int],
                 pattern: str = "*.nii",
                 image_suffix: str = ".nii",
                 mask_suffix: str = "_mask.nii.gz"
                 ) -> None:
        """

        Parameters
        ----------
        path: str
            Path to where the data is.

        labels: list[int]
            The list of labels used in masked image.

        pattern: str
            The pattern to use to `glob` to find the data in that directory.

        image_suffix: str
            The suffix to be replaced by the mask suffix to find a corresponding mask.

        mask_suffix: str
            The suffix to use to find the mask in the directory corresponding to an image.
        """
        self._path = path
        self._pattern = pattern
        self._image_suffix = image_suffix
        self._mask_suffix = mask_suffix
        self._labels = labels
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
        """

        Returns
        -------
        str
            The pattern.
        """
        return self._pattern


    @property
    def image_suffix(self) -> str:
        """

        Returns
        -------
        str
            The image suffix.
        """
        return self._image_suffix


    @property
    def mask_suffix(self) -> str:
        """

        Returns
        -------
        str
            The mask suffix.
        """
        return self._mask_suffix

    def _build_image_list(self) -> None:
        """
        Build list of images in the folder that the `path` property points to
        using the `pattern` property and the `glob` function.
        """
        self._image_list = glob(os.path.join(self._path, self._pattern))

    def __len__(self) -> int:
        """
        Output number of images in the dataset.

        Returns
        -------
        int
            The number of images in the path that fit the pattern.
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
        seg_fn = image_fn.replace(self._image_suffix, self._mask_suffix)

        self.reader.SetFileName(seg_fn)
        self.reader.Update()
        seg = vtkImageData_to_numpy(self.reader.GetOutput())

        class_segs = []

        for c in self._labels:
            class_segs.append((seg == c).astype(int))

        masks = np.stack(class_segs, axis=0)

        return image, masks
