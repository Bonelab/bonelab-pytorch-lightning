from __future__ import annotations
import torch

from blpytorchlightning.models.UNet2D import UNet2D


def get_torch_module():
    unet_spec = {
        "input_channels": 1,
        "output_classes": 3,
        "num_filters": [8, 16],
        "channels_per_group": 4,
        "dropout": 0.3
    }
    return UNet2D(**unet_spec)


def get_embedding_conversion_function() -> Callable[[torch.Tensor], torch.Tensor]:
    def embedding_conversion_function(x: torch.Tensor) -> torch.Tensor:
        return x

    return embedding_conversion_function


def get_loss_function() -> Callable[[torch.Tensor], torch.Tensor]:
    def loss_function(x: torch.Tensor) -> torch.Tensor:
        return x

    return loss_function
