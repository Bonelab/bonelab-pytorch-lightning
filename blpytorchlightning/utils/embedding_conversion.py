from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable


def create_approximate_heaviside(
    epsilon: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory function for the approximate heaviside function.

    Parameters
    ----------
    epsilon : float
        Scaling parameter for the approximate heaviside function. Controls how quickly likelihoods approach 0 or 1
        as input values increase or decrease.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        A function that transforms Tensors from level-set embedding fields to scalar fields on the open interval (0,1)
        that correspond to how likely it is that a voxel is inside the embedded surface.

    """

    def approximate_heaviside(x: torch.Tensor) -> torch.Tensor:
        """
        A function that transforms Tensors from level-set embedding fields to scalar fields on the open interval (0,1)
        that correspond to how likely it is that a voxel is inside the embedded surface.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor, should be a level-set embedding field.

        Returns
        -------
        torch.Tensor
            A tensor with float values on the open interval (0,1) values indicate how likely a voxel is to be within
            the surface embedded in the input embedding.
        """
        return 1 / 2 + (1 / np.pi) * torch.atan(x / epsilon)

    return approximate_heaviside


def create_convert_embeddings_to_predictions(
    epsilon: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory function for the embedding to predictions conversion function.

    Parameters
    ----------
    epsilon : float
        Scaling parameter for the approximate heaviside function. Controls how quickly likelihoods approach 0 or 1
        as input values increase or decrease.

    Returns
    -------
    Callable[[torch.Tensor],torch.Tensor]
        A function that takes a Tensor that is specifically two level-set embeddings (periosteal, endosteal) and
        transforms it to a Tensor with three channels: likelihood of being in each compartment (cortical,
        trabecular, background)
    """
    heaviside = create_approximate_heaviside(epsilon)

    def convert_embeddings_to_predictions(embeddings: torch.Tensor) -> torch.Tensor:
        """
        A function that takes a Tensor that is specifically two level-set embeddings (periosteal, endosteal) and
        transforms it to a Tensor with three channels: likelihood of being in each compartment (cortical,
        trabecular, background)

        Parameters
        ----------
        embeddings : torch.Tensor
            A Tensor with two channels: periosteal and endosteal level-set surface embeddings.

        Returns
        -------
        torch.Tensor
            A Tensor with three channels: likelihoods of being in the cortical, trabecular, background compartments.
        """
        phi_peri, phi_endo = embeddings[:, 0, :, :], embeddings[:, 1, :, :]
        # convert surface embeddings into voxel-wise class predictions
        pred_cort = heaviside(-phi_peri) * heaviside(phi_endo)
        pred_trab = heaviside(-phi_endo)
        pred_back = heaviside(phi_peri)
        # stack predictions, apply LogSoftmax to get log-probabilities
        preds = torch.stack((pred_cort, pred_trab, pred_back), dim=1)
        preds = nn.LogSoftmax(dim=1)(preds)
        return preds

    return convert_embeddings_to_predictions
