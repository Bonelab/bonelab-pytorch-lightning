from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc

from blpytorchlightning.utils.spatial_derivative_kernels import (
    get_2d_first_derivative_kernels,
    get_2d_second_derivative_kernels,
    get_3d_first_derivative_kernels,
    get_3d_second_derivative_kernels
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


class CurvatureLoss3D(nn.Module):
    """
    The 3D curvature (of the surface represented by a level-set surface embedding) loss function.
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
        self.kernels, self.denominators = {}, {}
        for k, v in get_3d_first_derivative_kernels().items():
            self.kernels[k] = v
            self.denominators[k] = vox_width
        for k, v in get_3d_second_derivative_kernels().items():
            self.kernels[k] = v
            self.denominators[k] = vox_width ** 2

    def _calculate_mean_curvature(
            self,
            gradient: list[torch.Tensor],
            hessian: list[list[torch.Tensor]],
            mag_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Internal function for calculating mean curvature field, given gradient and hessian fields.

        Parameters
        ----------
        gradient : list[torch.Tensor]
            List of gradient component fields

        hessian : list[list[torch.Tensor]]
            List of lists of hessian component fields

        mag_grad : torch.Tensor
            The magnitude of the gradient field

        Returns
        -------
        torch.Tensor
            The mean curvature field.
        """
        # init to 0
        mean_curvature = 0

        # add first term
        for i in range(3):
            for j in range(3):
                if i != j:
                    mean_curvature += torch.pow(gradient[i], 2) * hessian[j][j]

        # add second term
        for i in range(3):
            for j in range(i + 1, 3):
                mean_curvature -= 2 * gradient[i] * gradient[j] * hessian[i][j]

        # divide by the denominator
        return mean_curvature / (torch.pow(mag_grad, 3) + self.eps)

    def _calculate_gaussian_curvature(
            self,
            gradient: list[torch.Tensor],
            hessian: list[list[torch.Tensor]],
            mag_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Internal function for calculating Gaussian curvature field, given gradient and hessian fields.

        Parameters
        ----------
        gradient : list[torch.Tensor]
            List of gradient component fields

        hessian : list[list[torch.Tensor]]
            List of lists of hessian component fields

        mag_grad : torch.Tensor
            The magnitude of the gradient field

        Returns
        -------
        torch.Tensor
            The Gaussian curvature field.
        """
        # first term is the laplacian (trace of hessian) divided by mag grad
        laplace_over_mag_grad = 0
        for i in range(3):
            laplace_over_mag_grad += hessian[i][i]
        laplace_over_mag_grad = laplace_over_mag_grad / (mag_grad + self.eps)

        # second term is the sum of the hessian off-diagonals multiplied by the
        # two corresponding components of the gradient, all divided by the
        # cube of the mag grad
        hessian_offdiags_and_gradients_over_mag_grad_cubed = 0
        for i in range(3):
            for j in range(3):
                hessian_offdiags_and_gradients_over_mag_grad_cubed += gradient[i] * gradient[j] * hessian[i][j]
        hessian_offdiags_and_gradients_over_mag_grad_cubed = (
                hessian_offdiags_and_gradients_over_mag_grad_cubed / (torch.pow(mag_grad, 3) + self.eps)
        )

        return laplace_over_mag_grad - hessian_offdiags_and_gradients_over_mag_grad_cubed

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

        # do a little semantic reorganization into gradient and hessian structures to make the implementation
        # of the equations look nicer. these tensors are placed into these lists by reference so there should
        # be minimal performance impact...

        gradient = [grads["ddx"], grads["ddy"], grads["ddz"]]
        hessian = [
            [grads["d2dx2"], grads["d2dxdy"], grads["d2dxdz"]],
            [grads["d2dxdy"], grads["d2dy2"], grads["d2dydz"]],
            [grads["d2dxdz"], grads["d2dydz"], grads["d2dz2"]]
        ]

        # magnitude of the gradient is a useful quantity so calculate it ahead of time

        mag_grad = 0
        for i in range(3):
            mag_grad += torch.pow(gradient[i], 2)
        mag_grad = torch.sqrt(mag_grad + self.eps)

        # step 1: compute mean curvature
        mean_curvature = self._calculate_mean_curvature(gradient, hessian)

        # step 2: compute gaussian curvature
        gaussian_curvature = self._calculate_gaussian_curvature(gradient, hessian)

        # step 3: compute maximum principal curvature
        max_principal_curvature = (
            mean_curvature
            + torch.sqrt(
                torch.abs(torch.pow(mean_curvature, 2) - gaussian_curvature)
                + self.eps
            )
        )

        # then take the square of the curvature field divided by the curvature
        # threshold, subtract 1 from that, and apply a rectified linear unit.
        # we do this because we do not want there to be any penalty for curvature
        # that is within acceptable bounds, only excess curvature
        max_principal_curvature_relu = F.relu(
            torch.pow(max_principal_curvature / (self.curvature_threshold + self.eps), 2) - 1
        )

        # detect the zero crossings in phi
        phi_zero = zero_crossings(phi)

        # the final operation is to compute the overall loss by calculating the
        # average curvature on the contour defined by the zero level set,
        # where the zero level set has been determined as the voxels where
        # the level set crosses zero

        return torch.sum(phi_zero * max_principal_curvature_relu) / (torch.sum(phi_zero) + self.eps)


