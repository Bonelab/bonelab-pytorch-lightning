import unittest

from blpytorchlightning.tasks.SeGANTask import SeGANTask
from tests.tasks.common import get_loss_function, get_torch_module


class TestSeGANTask(unittest.TestCase):
    """ Test that we can import the module and work with the class. """

    def test_can_instantiate(self):
        learning_rate = 1e-4
        _ = SeGANTask(
            get_torch_module(),
            [get_torch_module()],
            get_loss_function(),
            learning_rate
        )


if __name__ == '__main__':
    unittest.main()
