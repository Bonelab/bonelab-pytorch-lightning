from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np
import torch


class BaseTransformer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(
        self,
        sample: Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]],
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]:
        """
        A magic method that allows this function to be called as a function.

        Parameters
        ----------
        sample: Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]
            The input sample, could be a tuple of numpy arrays or pytorch tensors.

        Returns
        -------
        Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]
            The transformed sample, could be a tuple of numpy arrays or pytorch tensors.
        """
        pass
