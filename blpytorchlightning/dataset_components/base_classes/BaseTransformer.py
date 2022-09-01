from __future__ import annotations

from abc import ABCMeta, abstractmethod
import numpy as np
import torch


class BaseTransformer(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self,
                 sample: Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]
                 ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        A magic method that allows this function to be called as a function.

        Parameters
        ----------
        sample: Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]
            The input sample, could be a tuple of numpy arrays or pytorch tensors.

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]
            The transformed sample, could be a tuple of numpy arrays or pytorch tensors.
        """
        pass
