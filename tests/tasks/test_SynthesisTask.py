import unittest

from blpytorchlightning.tasks.SynthesisTask import SynthesisTask
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from tests.tasks.common import get_loss_function, get_torch_module, get_embedding_conversion_function


class TestSynthesisTask(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        learning_rate = 1e-4
        SynthesisTask(
            get_torch_module(),
            SegmentationTask(
                get_torch_module(),
                get_loss_function(),
                learning_rate
            ),
            get_loss_function(),
            learning_rate
        )


if __name__ == '__main__':
    unittest.main()