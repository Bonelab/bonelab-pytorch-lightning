from __future__ import annotations

import torch
import pytorch_lightning as ptl


class RegistrationTask(ptl.LightningModule):

    def __init__(self):
        super().__init__()

    def training_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        pass

    def validation_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[torch.Tensor]:
        pass

    def test_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[torch.Tensor]:
        pass

    def predict_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> AdamW:
        """
        Required method, must return an optimizer for use in training.

        Returns
        -------
        AdamW
        """
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def _basic_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ) -> tuple[torch.Tensor, Optional[dict[torch.Tensor]]]:
        pass

    



