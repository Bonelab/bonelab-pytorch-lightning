import torch

# constant to prevent divide-by-zero errors
EPS = 1e-8


def dice_similarity_coefficient(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Dice similarity coefficient (binary, non-differentiable).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted segmentation / mask.

    target : torch.Tensor
        The target segmentation / mask.

    Returns
    -------
    torch.Tensor
        Tensor with a single value, the dice similarity coefficient of the two masks.
    """
    return (2 * torch.sum((prediction > 0) & (target > 0))) / (
        torch.sum(prediction > 0) + torch.sum(target > 0) + EPS
    )
