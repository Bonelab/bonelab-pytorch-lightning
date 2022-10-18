from __future__ import annotations

import torch
import pytorch_lightning as ptl
from torch.optim import AdamW
from typing import Callable

from blpytorchlightning.utils.error_metrics import dice_similarity_coefficient


class SeGANTask(ptl.LightningModule):
    """
    A pytorch-lightning task for image segmentation with a GAN. Requires a segmentor and a discriminator, and trains
    them in an alternating fashion (alternates every training step).
    Reference: https://doi.org/10.1007/s12021-018-9377-x
    Adds dice similarity coefficient to the metrics that are computed during training.
    """

    def __init__(
        self,
        segmentor: torch.nn.Module,
        discriminators: list[torch.nn.Module],
        loss_function: Callable[[torch.Tensor], torch.Tensor],
        learning_rate: float,
        logger: bool = True,
        log_on_step: bool = True,
        log_on_epoch: bool = True,
        log_sync_dist: bool = True
    ) -> None:
        """
        Initialization method.

        Parameters
        ----------
        segmentor : torch.nn.Module
            A pytorch module for segmenting images. Should take an image and return a segmentation.

        discriminators : list[torch.nn.Module]
            A list of pytorch modules for generating useful multi-scale feature maps from segmentation-masked images.
            Should take an image and return a list of pytorch tensors with feature maps at several levels of resolution.

        loss_function : Callable[[torch.Tensor], torch.Tensor]
            The loss function to optimize. Takes two multi-scale feature maps generated by the discriminator from the
            ground-truth and segmentor-generated segmentations, respectively. Minimized to train the segmentor and
            maximized to train the discriminator.

        learning_rate : float
            The learning rate to pass to the optimizer.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['segmentor', 'discriminators'])
        self.segmentor = segmentor
        self.discriminators = torch.nn.ModuleList(discriminators)
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.log_logger = logger
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.log_sync_dist = log_sync_dist

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        """
        Training step method. Negate the loss value if the discriminators are being trained.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A tuple containing the inputs and targets for a training step.

        batch_idx : int
            The index of the batch in the dataset. Not used in this method but must be accepted as an argument
            since pytorch-lightning's Trainers will pass it in during training.

        optimizer_idx : int
            The index into the collection of optimizers in the module that indicates which optimizer
            is currently being used. In this task, the segmentor and discriminators have separate optimizers.

        Returns
        -------
        torch.Tensor
            The loss value from the training step, with the graph attached for backprop.

        """
        stage = f"train_opt{optimizer_idx}"
        loss, _ = self._basic_step(batch, batch_idx, stage)

        if optimizer_idx == 0:
            # training the segmentor
            pass  # for now, don't do anything special

        if optimizer_idx == 1:
            # training the discriminators
            loss *= -1  # negate the loss since the discriminators want to maximize the differences

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
        y_hat, loss = self._compute_segmentation_and_loss(x, y)
        metrics = {
            f"{stage}_loss": loss.detach(),
            **self._get_dsc_metrics(y_hat, y, stage),
        }
        self.log_dict(
            metrics,
            on_step=self.log_on_step,
            on_epoch=self.log_on_epoch,
            logger=self.log_logger,
            sync_dist=self.log_sync_dist
        )
        return loss, metrics

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
        return self.segmentor(x)

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
        return self.segmentor(x)

    def configure_optimizers(self) -> tuple[AdamW, AdamW]:
        """
        Required method, must return an optimizer for use in training.
        Parameter group 0 is the segmentor, group 1 is the discriminators

        Returns
        -------
        tuple[AdamW, AdamW]
        """
        segmentor_optimizer = AdamW(self.segmentor.parameters(), lr=self.learning_rate)
        discriminators_optimizer = AdamW(
            self.discriminators.parameters(), lr=self.learning_rate
        )
        return segmentor_optimizer, discriminators_optimizer

    def _compute_segmentation_and_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The basic segmentation step method used by all the other step methods.
        Segments an image, calculates the multi-scale features from the image masked with the true segmentation
        and the predicted segmentation, and calculates the multi-scale MSE loss between features.

        Parameters
        ----------
        x: torch.Tensor
            Input image.

        y: torch.Tensor
            Ground-truth segmentation.

        Returns
        -------
        tuple[torch.Tensor, Optional[dict[torch.Tensor]]]
            The predicted segmentation and the L1 loss from the multi-scale feature maps.
        """
        y_hat = self.segmentor(x)
        loss = 0
        for i, d in enumerate(self.discriminators):
            x_true_masked = (y == i).unsqueeze(1) * x
            x_pred_masked = y_hat[:, [i], :, :] * x
            true_features = d(x_true_masked)
            pred_features = d(x_pred_masked)
            for tf, pf in zip(true_features, pred_features):
                loss += self.loss_function(pf, tf) / (
                    len(true_features) * len(self.discriminators)
                )
        return y_hat, loss

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
        metrics = {}
        num_classes = y_hat.shape[1]
        y_hat = torch.argmax(y_hat, dim=1)
        for c in range(num_classes):
            dsc = dice_similarity_coefficient(y == c, y_hat == c)
            metrics[f"{stage}_dsc_{c}"] = dsc
        return metrics
