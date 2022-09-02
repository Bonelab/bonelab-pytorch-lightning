from __future__ import annotations

from abc import ABCMeta, abstractmethod
import numpy as np


class BaseFileLoader(metaclass=ABCMeta):
    """
    Abstract base class for classes that are intended to load data.
    """

    @property
    @abstractmethod
    def path(self) -> str:
        """
        A getter method for the `path` property. `path` should point to the location of the data on disk.

        Returns
        -------
        str
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        A magic method to return how many samples are in the dataset. Necessary to function as iterator.

        Returns
        -------
        int
            Number of samples in dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        A magic method to return a sample. Samples should be tuples where the first element is the input and the
        second element is the target. Necessary to function as an iterator.

        Parameters
        ----------
        idx: int

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A sample, consisting of an input and target pair.
        """
        pass
