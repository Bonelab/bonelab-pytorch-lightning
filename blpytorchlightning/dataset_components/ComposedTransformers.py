from __future__ import annotations

from blpytorchlightning.dataset_components.base_classes.BaseTransformer import BaseTransformer


class ComposedTransformers(BaseTransformer):
    """ Convenience class for composing a list of transformations inheriting from BaseTransformer. """

    def __init__(self, transformers: List[BaseTransformer]) -> None:
        """
        Initialization method.

        Parameters
        ----------
        transformers : List[BaseTransformer]
            A list of transformers that will be applied sequentially to input images.
        """
        self.transformers = transformers

    def __call__(self,
                 sample: Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]
                 ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Function call magic method.

        Parameters
        ----------
        sample: Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]
            The input sample, could be a tuple of numpy arrays or pytorch tensors.

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]
            The transformed sample, could be a tuple of numpy arrays or pytorch tensors.
        """
        for transformer in self.transformers:
            sample = transformer(sample)
        return sample
