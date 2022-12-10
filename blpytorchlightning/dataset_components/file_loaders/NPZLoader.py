from __future__ import annotations

import numpy as np
import os

from blpytorchlightning.dataset_components.file_loaders.BaseFileLoader import (
    BaseFileLoader,
)


class NPZLoader(BaseFileLoader):
    """Class for loading data stored in npz files"""

    def __init__(
        self,
        path: str,
        pattern: str,
        inputs_keys: List[str],
        targets_keys: List[str],
        inputs_type: type = None,
        targets_type: type = None,
        binarize_targets: bool = False,
        add_null_class_to_targets: bool = False,
    ) -> None:
        """
        Initialization method.

        Parameters
        ----------
        path : str
            The path to the directory containing the data.

        pattern : str
            The pattern to use to `glob` to find the data in the `path directory`

        inputs_keys : List[str]
            A list of keys to use to index into the npz files to get the arrays that contain the input channels.
            The input tensor will be created with these arrays stacked on the channel axis in the order that
            the keys are given in this list.

        targets_keys : List[str]
            A list of keys to use to index into the npz files to get the arrays that contain the target channels.
            The target tensor will be created with these arrays stacked on the channel axis in the order that
            the keys are given in this list.

        inputs_type : type
            Type to cast the inputs numpy array to. If `None`, do not cast. Default: `None`

        targets_type : type
            Type to cast the targets numpy array to. If `None`, do not cast. Default: `None`

        binarize_targets : bool
            Boolean flag, set this to `True` to binarize values in the targets array. Defaults to `False`

        add_null_class_to_targets : bool
            A boolean flag, set this to `True` if you don't have a null class in your targets, and you want a
            null class to be added that is True anywhere that all other targets are False.
            Defaults to `False`.

        """
        self._path = path
        self._pattern = pattern
        self._inputs_keys = inputs_keys
        self._targets_keys = targets_keys
        self._inputs_type = inputs_type
        self._targets_type = targets_type
        self._binarize_targets = binarize_targets
        self._add_null_class_to_targets = add_null_class_to_targets
        self._build_image_list()

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

    @property
    def inputs_keys(self) -> List[str]:
        """

        Returns
        -------
        List[str]
            The `inputs_keys` property
        """
        return self._inputs_keys

    @property
    def targets_keys(self) -> List[str]:
        """

        Returns
        -------
        List[str]
            The `targets_keys` property
        """
        return self._targets_keys

    @property
    def inputs_type(self) -> type:
        """

        Returns
        -------
        type
            The `inputs_type` property
        """
        return self._inputs_type

    @property
    def targets_type(self) -> type:
        """

        Returns
        -------
        type
            The `targets_type` property
        """
        return self._targets_type

    @property
    def binarize_targets(self) -> bool:
        """

        Returns
        -------
        bool
            The `binarize_targets` property.
        """
        return self._binarize_targets

    @property
    def add_null_class_to_targets(self) -> bool:
        """

        Returns
        -------
        bool
            The `add_null_class_to_targets` property
        """
        return self._add_null_class_to_targets

    def _build_data_list(self) -> None:
        """
        Build list of images in the folder that the `path` property points to
        using the `pattern` property and the `glob` function.
        """
        self._data_list = glob(os.path.join(self._path, self._pattern))

    def __len__(self) -> int:
        """
        Output number of images in the dataset.

        Returns
        -------
        int
            The number of images in the path that fit the pattern.
        """
        return len(self._data_list)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        idx : int
            The index into the list of images in the dataset.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The first array is the inputs array while the second is the targets array.
            Dimensions depend on the data being loaded. They will be of shape (Ni,...) and (Nt,...)
            where Ni is the length of `inputs_keys` and Nt is the length of `targets_keys`
        """
        data = np.load(self._data_list[idx])
        inputs = np.stack([data[k] for k in self._inputs_keys])
        targets = np.stack([data[k] for k in self._targets_keys])
        if self._add_null_class_to_targets:
            targets = np.concatenate([targets, ~np.any(targets, axis=0, keepdims=True)])
        if self._inputs_type is not None:
            inputs = inputs.astype(self._inputs_type)
        if self._binarize_targets:
            targets = targets > 0
        if self._targets_type is not None:
            targets = targets.astype(self._targets_type)
        return inputs, targets

