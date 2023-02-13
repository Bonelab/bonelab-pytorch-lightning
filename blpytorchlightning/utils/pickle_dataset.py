from __future__ import annotations

import pickle
import yaml

from torch.utils.data import Dataset
from typing import Optional


def pickle_dataset(
        dataset: Dataset, folder: str, idxs: list[int], num_epochs: int, args: Optional[dict] = None
) -> None:
    """
    Pickle samples from a dataset and save to file to be consumed later.

    Parameters
    ----------
    dataset : Dataset
        The dataset to pickle.

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
            sample = dataset[idx]
            sample_fn = os.path.join(folder, subfolder, f"epoch_{e:d}.pickle")
            with open(sample_fn, "wb") as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
