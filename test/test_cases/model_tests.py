import unittest
import numpy as np
from src.model.Cell import Cell
from src.model.RNN import RNN

import pycuda.gpuarray

class RNNTestCase(unittest.TestCase):

    """def test_init(self):
        rnn = RNN(10, [1, 3, 5])
        answer = "5 layers: \n"
        answer += "  Input layer of size 10 to 10\n"
        answer += "  Hidden layer of size 10 to 1\n"
        answer += "  Hidden layer of size 1 to 3\n"
        answer += "  Hidden layer of size 3 to 5\n"
        answer += "  Output layer of size 5 to 10\n"

        self.assertEqual(rnn.__repr__(), answer)
    """

class CellTestCase(unittest.TestCase):

    """def test_forward_prop(self):
        cell = Cell(4, 4)
        x_t = np.array([[0], [0], [0], [1]])
        y_t = [[1], [1], [1], [1]]

        self.assertListEqual(cell.forward_prop(x_t).tolist(), y_t)
    """

    def test_forward_prop_gpu(self):
        cell = Cell(10, 2, (0,0))
        gpu_cell = Cell(10, 2, (0,0))
        gpu_cell.cell_to_gpu()
        x_t = np.zeros((2, 10))
        x_t[0][5] = 1
        x_t[1][7] = 1
        x_t_gpu = pycuda.gpuarray.to_gpu(x_t)

        h, c = cell.forward_prop(0, 0, x_t)
        h_gpu, c_gpu = gpu_cell.forward_prop_gpu(0, 0, x_t_gpu)

        self.assertListEqual(h.tolist(), h_gpu.get().tolist())