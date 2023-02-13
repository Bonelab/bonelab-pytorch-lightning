from __future__ import annotations

import torch
from typing import Callable, Optional

from blpytorchlightning.tasks.SeGANTask import SeGANTask


class SeGANEmbeddingTask(SeGANTask):
    """
    A pytorch-lightning task for image segmentation with a GAN. Requires a segmentor and a discriminator, and trains
    them in an alternating fashion (alternates every training step). The segmentor predicts level-set embeddings
    that are converted to probabilistic segmentations.
    Reference: https://doi.org/10.1007/s12021-018-9377-x
    Adds dice similarity coefficient to the metrics that are computed during training.
    """

    def __init__(
        self,
        segmentor: torch.nn.Module,
        discriminators: list[torch.nn.Module],
        embedding_conversion_function: Callable[[torch.Tensor], torch.Tensor],
        classification_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        curvature_loss_function: Callable[[torch.Tensor], torch.Tensor],
        maggrad_loss_function: Callable[[torch.Tensor], torch.Tensor],
        learning_rate: float,
        lambda_curvature: float = 1e-3,
        lambda_maggrad: float = 1e-3,
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

        embedding_conversion_function : Callable[[torch.Tensor], torch.Tensor]

        classification_loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The classification loss function to optimize.
            Takes a segmentation and ground truth and returns a loss value.

        curvature_loss_function : Callable[[torch.Tensor], torch.Tensor]
            Curvature regularization function. Takes a level-set embedding and returns a loss value.

        maggrad_loss_function : Callable[[torch.Tensor], torch.Tensor]
            Magnitude gradient regularization function. Takes a level-set embedding and returns a loss value.

        learning_rate : float
            The learning rate to pass to the optimizer.

        lambda_curvature : float
            Regularization coefficient for the curvature loss function.

        lambda_maggrad : float
            Regularization coefficient for the magnitude gradient egularization loss function.
        """
        super().__init__(segmentor, discriminators, None, learning_rate)
        self.embedding_conversion_function = embedding_conversion_function
        self.loss_functions = {
            "classification": classification_loss_function,
            "curvature": curvature_loss_function,
            "maggrad": maggrad_loss_function,
        }
        self.lambdas = {"curvature": lambda_curvature, "maggrad": lambda_maggrad}

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
        return self.embedding_conversion_function(self.segmentor(x))

    def forward_embeddings(self, x):
        """
        Forward pass embeddings method. Takes input image and returns level-set embeddings.

        Parameters
        ----------
        x : torch.Tensor
            An input image to segment.

        Returns
        -------
        torch.Tensor
            The level-set embeddings of the surfaces of the segmentation of the input image.

        """
        return self.segmentor(x)

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
        loss, _ = self._basic_step(batch, batch_idx, stage, optimizer_idx)

        return loss

    def _basic_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str,
        optimizer_idx: int = 0
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

        optimizer_idx : int
            The index of the optimizer we are using in this minibatch

        Returns
        -------
        tuple[torch.Tensor, Optional[dict[torch.Tensor]]]
            The first element of the tuple is the loss value with the graph attached for backprop. The second
            element of the tuple is the metrics dictionary.
        """
        loss_dict = {}
        x, y = batch
        phi = self.model(x)
        loss_dict["curvature"] = 0
        loss_dict["maggrad"] = 0
        for i in range(phi.shape[1]):
            loss_dict["curvature"] += self.loss_functions["curvature"](phi[:, [i], ...])
            loss_dict["maggrad"] += self.loss_functions["maggrad"](phi[:, [i], ...])
        y_hat = self.embedding_conversion_function(phi)
        loss_dict["classification"] = 0
        for i, d in enumerate(self.discriminators):
            x_true_masked = (y == i).unsqueeze(1) * x
            x_pred_masked = y_hat[:, [i], :, :] * x
            true_features = d(x_true_masked)
            pred_features = d(x_pred_masked)
            for tf, pf in zip(true_features, pred_features):
                loss_dict["classification"] += (
                    self.loss_functions["classification"](pf, tf)
                    / (len(true_features) * len(self.discriminators))
                )
        if optimizer_idx == 1:
            # training the discriminators
            loss_dict["classification"] *= -1
            # negate the loss since the discriminators want to maximize the differences
        metrics = {}
        for k, v in loss_dict.items():
            metrics[f"{stage}_{k}_loss"] = v
        metrics = {**metrics, **self._get_dsc_metrics(y_hat, y, stage)}
        self.log_dict(
            metrics,
            on_step=self.log_on_step,
            on_epoch=self.log_on_epoch,
            logger=self.log_logger,
            sync_dist=self.log_sync_dist,
        )
        loss = 0
        for v in loss_dict.values():
            loss += v
        return loss, metrics
