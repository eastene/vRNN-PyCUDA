import unittest
import numpy as np
from src.model.Layer import Layer
from src.model.RNN import RNN

class RNNTestCase(unittest.TestCase):

    def test_init(self):
        rnn = RNN(10, [1, 3, 5])
        answer = "5 layers: \n"
        answer += "  Input layer of size 10 to 10\n"
        answer += "  Hidden layer of size 10 to 1\n"
        answer += "  Hidden layer of size 1 to 3\n"
        answer += "  Hidden layer of size 3 to 5\n"
        answer += "  Output layer of size 5 to 10\n"

        self.assertEqual(rnn.__repr__(), answer)


class LayerTestCase(unittest.TestCase):

    def test_forward_prop(self):
        layer = Layer(4, 4)
        x_t = np.array([[0], [0], [0], [1]])
        y_t = [[1], [1], [1], [1]]

        self.assertListEqual(layer.forward_prop(x_t).tolist(), y_t)

    """def test_forward_prop_gpu(self):
        layer = Layer(10, 10)
        layer.layer_to_gpu()
        x_t_gpu = pycuda.gpuarray.zeros((10,1), np.float64)
        y_t = np.array([0 for i in range(10)]).transpose()

        self.assertListEqual(layer.forward_prop_gpu(x_t_gpu).tolist(), y_t.tolist())
    """