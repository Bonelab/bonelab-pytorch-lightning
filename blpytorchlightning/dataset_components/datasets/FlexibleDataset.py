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
            

        operations : Optional[List[Union[BaseSampler, BaseTransformer]]]
        """

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
