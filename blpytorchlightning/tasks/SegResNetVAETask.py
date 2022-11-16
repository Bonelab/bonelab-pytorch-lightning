from __future__ import annotations

import torch
import pytorch_lightning as ptl
from torch.optim import AdamW
from typing import Optional, Callable

from blpytorchlightning.utils.error_metrics import dice_similarity_coefficient


class SegResNetVAETask(ptl.LightningModule):
    """
    A very basic pytorch-lightning task meant for image segmentation. Adds dice similarity coefficient to the
    metrics that are computed during training. Can be used for 2D or 3D segmentation, with flexible inputs/outputs.
    Only compatible with models that take an image and return a segmentation with no extra fancy stuff required for
    training / processing. Should work for UNets and Vision Transformers. Won't work for modified models that predict
    level-set embeddings. Probably won't work for GANs.

    Child tasks can inherit from this one and define their own `_basic_step` and `forward` methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        learning_rate: float,
        logger: bool = True,
        log_on_step: bool = True,
        log_on_epoch: bool = True,
        log_sync_dist: bool = True,
    ) -> None:
        """
        Initialization method.

        Parameters
        ----------
        model : torch.nn.Module
            The model to train for the task. Should take an image and produce a segmentation.

        loss_function : Callable[[torch.Tensor], torch.Tensor]
            The loss function to optimize. Takes a segmentation and returns a loss value.

        learning_rate : float
            The learning rate to pass to the optimizer.

        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "loss_function"]
        )
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.log_logger = logger
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.log_sync_dist = log_sync_dist

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step method.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A tuple containing the inputs and targets for a training step.

        batch_idx : int
            The index of the batch in the dataset. Not used in this method but must be accepted as an argument
            since pytorch-lightning's Trainers will pass it in during training.

        Returns
        -------
        torch.Tensor
            The loss value from the training step, with the graph attached for backprop.

        """
        loss, _ = self._basic_step(batch, batch_idx, "train")
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[torch.Tensor]:
        """
        Validation step method.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A tuple containing the inputs and targets for a validation step.

        batch_idx : int
            The index of the batch in the dataset. Not used in this method but must be accepted as an argument
            since pytorch-lightning's Trainers will pass it in during training.

        Returns
        -------
        dict[torch.Tensor]
            A dictionary of performance metrics.
        """
        _, metrics = self._basic_step(batch, batch_idx, "val")
        return metrics

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[torch.Tensor]:
        """
        Testing step method.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A tuple containing the inputs and targets for a test step.

        batch_idx : int
            The index of the batch in the dataset. Not used in this method but must be accepted as an argument
            since pytorch-lightning's Trainers will pass it in during training.

        Returns
        -------
        dict[torch.Tensor]
            A dictionary of performance metrics.
        """
        _, metrics = self._basic_step(batch, batch_idx, "test")
        return metrics

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Prediction step method.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A tuple containing the inputs and targets for a test step.

        batch_idx : int
            The index of the batch in the dataset. Not used in this method but must be accepted as an argument
            since pytorch-lightning's Trainers will pass it in during training.

        Returns
        -------
        torch.Tensor
            Predictions on the input portion of the batch inputs.
        """
        x, _ = batch
        y_hat, _ = self.model(x)
        return y_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method. Takes input image and returns segmentation.

        Parameters
        ----------
        x : torch.Tensor
            An input image to segment.

        Returns
        -------
        torch.Tensor
            The segmentation of the input image.
        """
        y_hat, _ = self.model(x)
        return y_hat

    def configure_optimizers(self) -> AdamW:
        """
        Required method, must return an optimizer for use in training.

        Returns
        -------
        AdamW
        """
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def _basic_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ) -> tuple[torch.Tensor, Optional[dict[torch.Tensor]]]:
        """
        The basic segmentation step method used by all the other step methods.
        Segments an image, returns loss and metrics.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A tuple containing the inputs and targets for a test step.

        batch_idx : int
            The index of the batch in the dataset. Not used in this method but must be accepted as an argument
            since pytorch-lightning's Trainers will pass it in during training.

        stage : str
            The stage of task training the task is currently in, e.g. "train" or "validate". Used for naming keys in
            the metrics dictionary.

        Returns
        -------
        tuple[torch.Tensor, Optional[dict[torch.Tensor]]]
            The first element of the tuple is the loss value with the graph attached for backprop. The second
            element of the tuple is the metrics dictionary.
        """
        x, y = batch
        y_hat, vae_loss = self.model(x)
        loss = self.loss_function(y_hat, y)
        metrics = {
            f"{stage}_loss": loss.detach(),
            f"{stage}_vae_loss": vae_loss.detach() if vae_loss is not None else 0,
            **self._get_dsc_metrics(y_hat, y, stage),
        }
        self.log_dict(
            metrics,
            on_step=self.log_on_step,
            on_epoch=self.log_on_epoch,
            logger=self.log_logger,
            sync_dist=self.log_sync_dist,
        )
        if vae_loss is not None:
            return loss + vae_loss, metrics
        else:
            return loss, metrics

    @staticmethod
    def _get_dsc_metrics(
        y_hat: torch.Tensor, y: torch.Tensor, stage: str
    ) -> dict[torch.Tensor]:
        """
        Static method for adding the dice similarity coefficient to the metrics dictionary.

        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted segmentation.

        y : torch.Tensor
            Target segmentation.

        stage : str
            The stage of task training the task is currently in, e.g. "train" or "validate". Used for naming keys in
            the metrics dictionary.

        Returns
        -------
        dict[torch.Tensor]
            A dictionary with the dice similarity coefficient values for each class in the segmentation.
        """
        num_classes = y_hat.shape[1]
        y_hat = torch.argmax(y_hat, dim=1)
        metrics = {}
        for c in range(num_classes):
            dsc = dice_similarity_coefficient(y == c, y_hat == c)
            metrics[f"{stage}_dsc_{c}"] = dsc
        return metrics
