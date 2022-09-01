import torch
import torch.nn as nn
import torch.nn.functional as torchfunc

from blpytorchlightning.utils.spatial_derivative_kernels import get_2d_first_derivative_kernels
from blpytorchlightning.utils.zero_crossings import zero_crossings


class MagnitudeGradientSDTLoss(nn.Module):
    """
    The magnitude gradient (of) Signed Distance Transform loss function.
    """

    def __init__(self, vox_width: float) -> None:
        """
        Initialization method.

        Parameters
        ----------
        vox_width : float
            The spatial width of a voxel in the images / surface embeddings. Needed to compute a physically meaningful
            gradient.
        """
        super().__init__()
        self.eps = 1e-8
        self.kernels = get_2d_first_derivative_kernels()
        self.vox_width = vox_width

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        The forward pass method - calculate the loss value from a surface embedding field.

        Parameters
        ----------
        phi : torch.Tensor
            A surface embedding field contained in a torch Tensor with float values.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        grads = {}
        for k, v in self.kernels.items():
            torch_kernel = torch.tensor(v, device=phi.device, dtype=phi.dtype).unsqueeze(0).unsqueeze(0)
            grads[k] = torchfunc.conv2d(phi, torch_kernel) / self.vox_width
        magnitude_gradient_of_phi = torch.sqrt(torch.pow(grads['ddx'], 2) + torch.pow(grads['ddy'], 2) + self.eps)
        squared_log_of_magnitude_gradient_of_phi = torch.pow(torch.log(magnitude_gradient_of_phi), 2)
        phi_not_zero = torch.logical_not(zero_crossings(phi))
        return torch.sum(phi_not_zero * squared_log_of_magnitude_gradient_of_phi) / (torch.sum(phi_not_zero) + self.eps)
