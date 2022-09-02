import torch
import pytorch_lightning as ptl
from torch.optim import AdamW
from blpytorchlightning.utils.error_metrics import dice_similarity_coefficient

# TODO: Rewrite this so that it just takes the segmentations as input, and have a separate script generate
#       the segmentations and save them ahead of time. Inefficient to compute the segmentations on-demand


class SynthesisTask(ptl.LightningModule):
    def __init__(
        self, synthesis_model, segmentation_task, loss_function, learning_rate
    ):
        super().__init__()
        self.synthesis_model = synthesis_model
        self.segmentation_task = segmentation_task
        self.segmentation_task.freeze()
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self._frozen_segmentation(x)
        y_hat = self.synthesis_model(z)
        loss = self.loss_function(y_hat, y)
        dsc = dice_similarity_coefficient(y_hat, y)
        metrics = {"train_loss": loss, "train_dsc": dsc}
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self._frozen_segmentation(x)
        y_hat = self.synthesis_model(z)
        loss = self.loss_function(y_hat, y)
        dsc = dice_similarity_coefficient(y_hat, y)
        metrics = {"val_loss": loss, "val_dsc": dsc}
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)
        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.synthesis_model(self._frozen_segmentation(x))
        loss = self.loss_function(y_hat, y)
        dsc = dice_similarity_coefficient(y_hat, y)
        metrics = {"test_loss": loss, "test_dsc": dsc}
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        z = self._frozen_segmentation(x)
        return self.synthesis_model(z)

    def forward(self, x):
        return self.synthesis_model(self._frozen_segmentation(x))

    def configure_optimizers(self):
        return AdamW(self.synthesis_model.parameters(), lr=self.learning_rate)

    def _frozen_segmentation_old(self, x):
        # x comes in with dims (B,1,Nx,Ny,Nz)
        # use the segmentation model to construct predictions slice by slice
        x_shape = list(x.shape)
        z_shape = x_shape.copy()
        z_shape[1] = 4  # we're going to create 4 channels instead of 1
        z = torch.zeros(z_shape, dtype=x.dtype, device=x.device)
        z[:, 0, ...] = x[:, 0, ...]  # first channel is just the image itself
        for c in [1, 2, 3]:
            x_slice = [slice(None)] * len(x_shape)
            z_slice = [slice(None)] * len(z_shape)
            x_slice[1], z_slice[1] = [0], [c]
            # take from the only channel of the raw image and send it to the
            # current channel, c, of the predictions tensor
            for i in range(x.shape[c + 1]):
                # now iterate through the "slices" of the tensors along the dim
                # corresponding to the current channel
                x_slice[c + 1], z_slice[c + 1] = i, i
                z[tuple(z_slice)] = self.segmentation_task(x[tuple(x_slice)])
        return z

    def _frozen_segmentation(self, x):
        # x comes in with dims (B,1,Nx,Ny,Nz)
        # use the segmentation model to construct predictions slice by slice
        x_shape = list(x.shape)
        z_shape = x_shape.copy()
        z_shape[1] = 4  # we're going to create 4 channels instead of 1
        z = torch.zeros(z_shape, dtype=x.dtype, device=x.device)
        z[:, 0, ...] = x[:, 0, ...]  # first channel is just the image itself
        for c in [1, 2, 3]:
            preds = self.segmentation_task(
                torch.cat(torch.split(x, 1, dim=(c + 1)), dim=0).squeeze(c + 1)
            )
            print(preds.shape)
            """
            x_slice = [slice(None)]*len(x_shape)
            z_slice = [slice(None)]*len(z_shape)
            x_slice[1], z_slice[1] = [0], [c]
            # take from the only channel of the raw image and send it to the
            # current channel, c, of the predictions tensor


            for i in range(x.shape[c+1]):
                # now iterate through the "slices" of the tensors along the dim
                # corresponding to the current channel
                x_slice[c+1], z_slice[c+1] = i, i
                z[tuple(z_slice)] = self.segmentation_task(x[tuple(x_slice)])
            """
        return z
