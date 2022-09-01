from __future__ import annotations
import torch

from blpytorchlightning.tasks.SegmentationTask import SegmentationTask


class SegmentationEmbeddingTask(SegmentationTask):
    """
    The segmentation task is slightly modified so that we are predicting level-set embeddings rather than classifying
    voxels directly.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            embedding_conversion_function: Callable[[torch.Tensor], torch.Tensor],
            classification_loss_function: Callable[[torch.Tensor], torch.Tensor],
            curvature_loss_function: Callable[[torch.Tensor], torch.Tensor],
            maggrad_loss_function: Callable[[torch.Tensor], torch.Tensor],
            learning_rate: float,
            lambda_curvature: float = 1e-3,
            lambda_maggrad: float = 1e-3
    ):
        """
        Initialization method.

        Parameters
        ----------
        model : torch.nn.Module
            The model to train for the task. Should take an image and produce a segmentation.

        embedding_conversion_function : Callable[[torch.Tensor], torch.Tensor]

        classification_loss_function : Callable[[torch.Tensor], torch.Tensor]
            The classification loss function to optimize. Takes a segmentation and returns a loss value.

        curvature_loss_function : Callable[[torch.Tensor], torch.Tensor]
            Curvature regularization function. Takes a level-set embedding and returns a loss value.

        maggrad_loss_function : Callable[[torch.Tensor], torch.Tensor]
            Magnitude gradient regularization function. Takes a level-set embedding and returns a loss value.

        learning_rate : float
            The learning rate to pass to the optimizer.

        lambda_curvature : float
            Regularization coefficient for the curvature loss function.

        lambda_maggrad : float
            Regularization coefficient for the magnitude gradient egularization loss function.
        """
        super().__init__(model, None, learning_rate)
        self.embedding_conversion_function = embedding_conversion_function
        self.loss_functions = {
            'classification': classification_loss_function,
            'curvature': curvature_loss_function,
            'maggrad': maggrad_loss_function
        }
        self.lambdas = {'curvature': lambda_curvature, 'maggrad': lambda_maggrad}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method. Takes input image and returns segmentation.

        Parameters
        ----------
        x : torch.Tensor
            An input image to segment.

        Returns
        -------
        torch.Tensor
            The segmentation of the input image.

        """
        return self.embedding_conversion_function(self.model(x))

    def forward_embeddings(self, x):
        """
        Forward pass embeddings method. Takes input image and returns level-set embeddings.

        Parameters
        ----------
        x : torch.Tensor
            An input image to segment.

        Returns
        -------
        torch.Tensor
            The level-set embeddings of the surfaces of the segmentation of the input image.

        """
        return self.model(x)

    def _basic_step(self,
                    batch: Tuple[torch.Tensor, torch.Tensor],
                    batch_idx: int,
                    stage: str
                    ) -> Tuple[torch.Tensor, Optional[Dict[Torch.Tensor]]]:
        """
        The basic segmentation step method used by all the other step methods.
        Segments an image, returns loss and metrics.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the inputs and targets for a test step.

        batch_idx : int
            The index of the batch in the dataset. Not used in this method but must be accepted as an argument
            since pytorch-lightning's Trainers will pass it in during training.

        stage : str
            The stage of task training the task is currently in, e.g. "train" or "validate". Used for naming keys in
            the metrics dictionary.

        Returns
        -------
        Tuple[torch.Tensor, Optional[Dict[Torch.Tensor]]]
            The first element of the tuple is the loss value with the graph attached for backprop. The second
            element of the tuple is the metrics dictionary.
        """
        loss_dict = {}
        x, y = batch
        phi = self.model(x)
        loss_dict['curvature'] = 0
        loss_dict['maggrad'] = 0
        for i in range(phi.shape[1]):
            loss_dict['curvature'] += self.loss_functions['curvature'](phi[:, [i], ...])
            loss_dict['maggrad'] += self.loss_functions['maggrad'](phi[:, [i], ...])
        y_hat = self.embedding_conversion_function(phi)
        classification_loss = self.loss_functions['classification'](y_hat, y)
        loss_dict['total'] = (
                classification_loss
                + self.lambdas['curvature'] * loss_dict['curvature']
                + self.lambdas['maggrad'] * loss_dict['maggrad']
        )
        metrics = {}
        for k, v in loss_dict.items():
            metrics[f'{stage}_{k}_loss'] = v
        metrics = {**metrics, **self._get_dsc_metrics(y_het, y, stage)}
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)
        return loss_dict['total'], metrics
