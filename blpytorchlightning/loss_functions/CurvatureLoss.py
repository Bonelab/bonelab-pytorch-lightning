import torch
import torch.nn as nn
import torch.nn.functional as torchfunc

from blpytorchlightning.utils.spatial_derivative_kernels import (
    get_2d_first_derivative_kernels,
    get_2d_second_derivative_kernels,
)
from blpytorchlightning.utils.zero_crossings import zero_crossings


class CurvatureLoss(nn.Module):
    """
    The curvature (of the surface represented by a level-set surface embedding) loss function.
    """

    def __init__(self, vox_width: float, curvature_threshold: float) -> None:
        """
        Initialization method.

        Parameters
        ----------
        vox_width : float
            Spatial width of a voxel in the image / embedding field. Needed for curvatures to be physically meaningful.

        curvature_threshold : float
            The threshold below which curvatures are not penalized and above which curvatures begin incurring cost.
            The units of 2D curvature are [1/L] and the units of `curvature_threshold` must be consistent with the
            units given for `vox_width`. e.g. um and 1/um or m and 1/m
        """
        super().__init__()
        self.eps = 1e-8
        self.curvature_threshold = curvature_threshold
        self.kernels = {
            **get_2d_first_derivative_kernels(),
            **get_2d_second_derivative_kernels(),
        }
        self.denominators = {
            "ddx": vox_width,
            "ddy": vox_width,
            "d2dx2": vox_width ** 2,
            "d2dy2": vox_width ** 2,
            "d2dxdy": vox_width ** 2,
        }

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method. Calculates the curvature loss value from an embedding field

        Parameters
        ----------
        phi : torch.Tensor
            Embedding field.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        grads = {}
        for k, v in self.kernels.items():
            torch_kernel = (
                torch.tensor(v, device=phi.device, dtype=phi.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            grads[k] = torchfunc.conv2d(phi, torch_kernel) / self.denominators[k]
        curvature_numerator = (
            grads["d2dxdy"] * torch.pow(grads["ddy"], 2)
            - 2 * grads["ddy"] * grads["ddx"] * grads["d2dxdy"]
            + grads["d2dxdy"] * torch.pow(grads["ddx"], 2)
        )
        curvature_denominator = torch.pow(
            torch.pow(grads["ddx"], 2) + torch.pow(grads["ddy"], 2), 3 / 2
        )
        curvature = curvature_numerator / (curvature_denominator + self.eps)
        curvature_relu = torchfunc.relu(
            torch.pow(curvature / (self.curvature_thresh + self.eps), 2) - 1
        )
        phi_zero = zero_crossings(phi)
        return torch.sum(phi_zero * curvature_relu) / (torch.sum(phi_zero) + self.eps)
