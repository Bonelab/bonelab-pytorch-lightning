from __future__ import annotations

import os
import numpy as np
import pickle
import torch
from typing import Optional, Union

from dataclasses import dataclass
from torch.utils.data import Dataset
from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)


@dataclass
class PickledData:
    """
    Class for tracking and loading pickled samples
    """

    path: str
    epoch: int = 0

    def load_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the pickle file and increment the epoch counter.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The sample.

        """
        sample_fn = os.path.join(self.path, f"epoch_{self.epoch:d}.pickle")
        if not os.path.exists(sample_fn):
            self.epoch = 0
            sample_fn = os.path.join(self.path, f"epoch_{self.epoch:d}.pickle")
            if not os.path.exists(sample_fn):
                raise FileNotFoundError(f"No pickle file for epoch 0 in {self.path}")
        with open(sample_fn, "rb") as f:
            sample = pickle.load(f)
        self.epoch += 1
        return sample

    def reset(self) -> None:
        """Reset back to epoch 0 for new training"""
        self.epoch = 0


class PickledDataset(Dataset):
    """
     A Dataset to load samples that have been loaded, sampled, and pickled to file to very fast loading. Uses the
    `PickledData` dataclass for file loading operations. Does not require a file loader or sampler, but can optionally
    be given a transformer (and should be, to convert samples to tensors at least).

    If you want to use this, you need to create your own Dataset by composition and then implement the `pickle_dataset`
    method to have the Dataset load, sample, and pickle your data. See `HRpQCTDataset` for an example.

    """

    def __init__(
        self, folder: str, transformer: Optional[BaseTransformer] = None
    ) -> None:
        """
        Initialization method

        Parameters
        ----------
        folder : str
            The directory on disk where the pickled sample subdirectories are located.

        transformer: BaseTransformer
            The transformation object to apply to loaded samples.

        """
        super().__init__()
        self._sample_list = [
            PickledData(f.path) for f in os.scandir(folder) if f.is_dir()
        ]
        self.transformer = transformer

    def __len__(self) -> int:
        """
        Length of dataset is the number of samples in the list

        Returns
        -------
        int

        """
        return len(self._sample_list)

    def __getitem__(
        self, idx: int
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]:
        """
        Use the PickledData class method `load_data` to return a sample

        Parameters
        ----------
        idx : int
            The index into the list of samples to load.

        Returns
        -------
        Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]
            The sample.

        """
        sample = self._sample_list[idx].load_sample()
        if self.transformer is not None:
            sample = self.transformer(sample)
        return sample

    def reset(self) -> None:
        """
        Reset all samples to epoch 0
        """
        for s in self._sample_list:
            s.reset()
