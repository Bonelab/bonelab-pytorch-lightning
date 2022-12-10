from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    arXiv:1606.04797v1 [cs.CV] 15 Jun 2016
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """
        Initialization method.
        """
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        p : torch.Tensor
            Predicted segmentation probabilities.
            Should be floats, and should be of shape: [N, num_classes, ...]

        y : torch.Tensor
            Target segmentation.
            Should be binary, and should be of shape [N, num_classes, ...]

        Returns
        -------
        torch.Tensor
            The Dice loss, averaged over all classes equally.
        """
        loss = 0
        for c in range(p.shape[1]):
            loss += 1 - (
                (2 * (y[:, c, ...] * p[:, c, ...]).sum())
                / (
                    torch.pow(y[:, c, ...], 2).sum()
                    + torch.pow(p[:, c, ...], 2).sum()
                    + self.eps
                )
            )
        return loss / p.shape[1]
