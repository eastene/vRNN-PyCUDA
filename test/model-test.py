import unittest
import numpy as np
from model.Layer import Layer

class LayerTest(unittest.TestCase):

    def test_forward_prop(self):
        layer = Layer(4, 4)
        x_t = np.array([[0], [0], [0], [1]])
        y_t = [[1], [1], [1], [1]]

        self.assertListEqual(layer.forward_prop(x_t).tolist(), y_t)