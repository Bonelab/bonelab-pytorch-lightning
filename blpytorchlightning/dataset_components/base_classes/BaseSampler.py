from __future__ import annotations

from abc import ABCMeta, abstractmethod
import numpy as np


class BaseSampler(metaclass=ABCMeta):
    """
    Abstract base class for classes that are intended to (sub)sample data.
    """

    @abstractmethod
    def __call__(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A magic method that allows this function to be called as a function. Must be implemented and must apply the
        sampling strategy to an input data sample to produce a new sample.

        Parameters
        ----------
        sample : tuple[np.ndarray, np.ndarray]
            The input sample, likely directly from the object that loads data from file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A new sample that is sub-sampled from the input sample.
        """
        pass
