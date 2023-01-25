import torch


def zero_crossings(x: torch.Tensor) -> torch.Tensor:
    """
    Detect zero-crossings in a pytorch tensor

    Parameters
    ----------
    x: torch.Tensor
        The input tensor, ideally should be a level-set embedding field. Shape (B,C,X1,...,Xn)

    Returns
    -------
    torch.Tensor
        Binary tensor with modified shape (B,C,X1-2,...,Xn-2). Zero-crossings cannot be detected on boundary.
    """
    sign_x = torch.sign(x)
    num_spatial_dims = x.dim() - 2  # two of the dims are batch, channels
    # if x = 0, obviously a zero-crossing
    z = (sign_x == 0)[tuple([slice(None), slice(None)] + [slice(1, -1) for _ in range(num_spatial_dims)])]
    # then, check anywhere the sign differs by 2 in adjacent voxels
    for d in num_spatial_dims:
        st_center = tuple([slice(None), slice(None)] + [slice(1, -1) for _ in range(num_spatial_dims)])
        st_lower = tuple(
            [slice(None), slice(None)]
            + [slice(None, -2) if i == d else slice(1, -1) for i in range(num_spatial_dims)]
        )
        st_greater = tuple(
            [slice(None), slice(None)]
            + [slice(2, None) if i == d else slice(1, -1) for i in range(num_spatial_dims)]
        )
        z += torch.abs(sign_x[st_center] - sign_x[st_lower]) == 2
        z += torch.abs(sign_x[st_center] - sign_x[st_greater]) == 2
    # return a binary mask, True anywhere a zero-crossing was detected
    return z > 0
