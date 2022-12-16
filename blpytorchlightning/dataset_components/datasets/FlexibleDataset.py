from __future__ import annotations

import numpy as np
import os
import pickle
import torch
import yaml
from torch.utils.data import Dataset
from typing import Optional, Union, List

from blpytorchlightning.dataset_components.file_loaders.BaseFileLoader import (
    BaseFileLoader,
)
from blpytorchlightning.dataset_components.samplers.BaseSampler import BaseSampler
from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)


class FlexibleDataset(Dataset):
    """
    A more flexible Dataset sub-class for medical images.
    Takes a single file loader and then any number of sampler and/or transform objects in any order.
    Making sure the samplers and transformations work in the order you have given them is up to you.
    """

    def __init__(
        self,
        file_loader: BaseFileLoader,
        operations: Optional[List[Union[BaseSampler, BaseTransformer]]] = None
    ) -> None:
        """

        Parameters
        ----------
        file_loader : BaseFileLoader
            Responsible for managing and loading data from file.

        operations : Optional[List[Union[BaseSampler, BaseTransformer]]]
            A list of sampling and/or transforming objects that will be applied to the data samples in the
            order they appear in the list.
        """
        super().__init__()
        self.file_loader = file_loader
        self.operations = operations

    def __len__(self):
        """
        A magic method to return how many samples are in the dataset. Necessary to function as iterator.
        The length of this dataset is just the length of the file_loader.

        Returns
        -------
        int
            Number of samples in dataset.
        """
        return len(self.file_loader)

    def __getitem__(self, idx: int) -> Union[dict, tuple]:
        sample = self.file_loader[idx]
        for operation in operations:
            sample = operation(sample)
        return sample
