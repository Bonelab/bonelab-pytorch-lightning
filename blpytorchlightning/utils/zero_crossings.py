import torch


def zero_crossings(x: torch.Tensor) -> torch.Tensor:
    """
    Detect zero-crossings in a pytorch tensor

    Parameters
    ----------
    x: torch.Tensor
        The input tensor, ideally should be a level-set embedding field. Shape (B,C,H,W)

    Returns
    -------
    torch.Tensor
        Binary tensor with modified shape (B,C,H-2,W-2). Zero-crossings cannot be detected on boundary.
    """
    sign_x = torch.sign(x)
    z = (sign_x == 0)[:, :, 1:-1, 1:-1]  # if x=0, obviously a zero-crossing
    # then, check anywhere the sign differs by 2 in adjacent voxels
    z += torch.abs(sign_x[:, :, 1:-1, 1:-1] - sign_x[:, :, 1:-1, 2:]) == 2
    z += torch.abs(sign_x[:, :, 1:-1, 1:-1] - sign_x[:, :, 1:-1, :-2]) == 2
    z += torch.abs(sign_x[:, :, 1:-1, 1:-1] - sign_x[:, :, 2:, 1:-1]) == 2
    z += torch.abs(sign_x[:, :, 1:-1, 1:-1] - sign_x[:, :, :-2, 1:-1]) == 2
    # return a binary mask, True anywhere a zero-crossing was detected
    return z > 0
