from __future__ import annotations

import numpy as np
import os
import pickle
import torch
import yaml
from torch.utils.data import Dataset
from typing import Optional, Union

from blpytorchlightning.dataset_components.base_classes.BaseFileLoader import (
    BaseFileLoader,
)
from blpytorchlightning.dataset_components.base_classes.BaseFileLoader import (
    BaseSampler,
)
from blpytorchlightning.dataset_components.base_classes.BaseFileLoader import (
    BaseTransformer,
)


class HRpQCTDataset(Dataset):
    """
    A Dataset class for HRpQCT images, constructed by composition.
    Given a loader, sampler, and (optional) transform objects, it implements the
    required methods to serve as a Dataset given to a Dataloader during model training.
    """

    def __init__(
        self,
        file_loader: BaseFileLoader,
        sampler: BaseSampler,
        transformer: Optional[BaseTransformer] = None,
    ) -> None:
        """
        Initialization method.

        Parameters
        ----------
        file_loader: BaseFileLoader
            Responsible for managing and loading data from file (or memory) and serving up individual samples as
            tuples of images and masks (inputs and targets) as numpy arrays.

        sampler: BaseSampler
            Responsible for sampling data from the raw loaded data. e.g. take a small patch, extract slices, etc.

        transformer: BaseTransformer
            Responsible for applying any final transformations to the data, such as augmentation, and for transferring
            the sample to a torch tensor. One of the transforms should convert the samples to tensors if you're going
            to use this object to load data on-demand for training a pytorch model/task.
        """
        super().__init__()
        self.file_loader = file_loader
        self.sampler = sampler
        self.transformer = transformer

    def __len__(self) -> int:
        """
        A magic method to return how many samples are in the dataset. Necessary to function as iterator.
        The length of this dataset is just the length of the file_loader.

        Returns
        -------
        int
            Number of samples in dataset.
        """
        return len(self.file_loader)

    def __getitem__(
        self, idx: int
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]:
        """
        A magic method to return a sample. Samples are tuples where the first element is the input and the
        second element is the target. Necessary to function as an iterator.

        Parameters
        ----------
        idx: int

        Returns
        -------
        Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]
            A sample, consisting of an input and target pair.
        """
        sample = self.sampler(self.file_loader[idx])
        if self.transformer:
            sample = self.transformer(sample)
        return sample

    def pickle_dataset(
        self, folder: str, idxs: list[int], num_epochs: int, args: Optional[dict] = None
    ) -> None:
        """
        Pickle samples from a dataset and save to file to be consumed later.

        Parameters
        ----------
        folder : str
            The directory to save the pickled samples to.

        idxs : list[int]
            The sample indices to pickle.

        num_epochs : int
            The number of epochs you plan to train for - a separate pickled sample will be taken from each image.

        args : Optional[dict]
            The configuration dictionary can be passed here. If given, it will be written to a yaml file in `folder`
            called `hparams.yaml`. Useful if you want to make sure the scripts that pickle and train use the exact
            same configuration.
        """
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        if args:
            with open(os.path.join(args.pickle_dir, "hparams.yaml"), "w") as f:
                yaml.dump(args.__dict__, f)
        for idx in idxs:
            subfolder = f"{idx:d}"
            if not os.path.isdir(os.path.join(folder, subfolder)):
                os.mkdir(os.path.join(folder, subfolder))
            for e in range(num_epochs):
                sample = self[idx]
                sample_fn = os.path.join(folder, subfolder, f"epoch_{e:d}.pickle")
                with open(sample_fn, "wb") as f:
                    pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
