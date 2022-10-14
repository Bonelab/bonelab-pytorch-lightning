from __future__ import annotations

import torch
import pytorch_lightning as ptl
from torch.optim import AdamW
from typing import Callable


class DiNTSSearchTask(ptl.LightningModule):
    """
    Pytorch-lightning task for searching for an optimal DiNTS topology for a given task.
    Based on: https://github.com/Project-MONAI/tutorials/blob/main/automl/DiNTS/search_dints.py
    """

    def __init__(self):
        super().__init__()

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def predict_step(self):
        pass

    def forward(self):
        pass

    def configure_optimizers(self):
        pass
