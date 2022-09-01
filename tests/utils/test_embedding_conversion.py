import torch
import unittest

from blpytorchlightning.utils.embedding_conversion import (
    create_approximate_heaviside,
    create_convert_embeddings_to_predictions
)


class TestEmbeddingConversion(unittest.TestCase):
    """ Test that we can import the module and call the functions. """

    def test_create_approximate_heaviside(self):
        epsilon = 0.1
        approximate_heaviside = create_approximate_heaviside(epsilon)
        approximate_heaviside(torch.Tensor(10))

    def test_create_convert_embeddings_to_predictions(self):
        epsilon = 0.1
        convert_embeddings_to_predictions = create_convert_embeddings_to_predictions(epsilon)
        convert_embeddings_to_predictions(torch.zeros(1, 2, 3, 3))
