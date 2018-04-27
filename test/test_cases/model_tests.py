import unittest
import numpy as np
from src.model.Cell import Cell
from src.model.LSTM import RNN

import pycuda.gpuarray
from pycuda.tools import mark_cuda_test

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

    @mark_cuda_test
    def test_forward_prop_gpu(self):
        vocab = 10
        batches = 2
        cell = Cell(vocab, batches, (0,0))
        gpu_cell = Cell(vocab, batches, (0,0))
        gpu_cell.cell_to_gpu()
        x_t = np.zeros((vocab, batches))
        h_prev = np.zeros((vocab, batches))
        c_prev = np.zeros((vocab, batches))
        x_t[5][0] = 1
        x_t[7][1] = 1
        x_t_gpu = pycuda.gpuarray.to_gpu(x_t)
        h_prev_gpu = pycuda.gpuarray.to_gpu(h_prev)
        c_prev_gpu = pycuda.gpuarray.to_gpu(c_prev)

        h, c = cell.forward_prop(c_prev, h_prev, x_t)
        print("CPU done")
        h_gpu, c_gpu = gpu_cell.forward_prop_gpu(c_prev_gpu, h_prev_gpu, x_t_gpu)
        gpu_cell.cell_from_gpu()
        print("GPU Done")

        h_gpu_mem = h_gpu.get()

        for j in range(batches):
            for i in range(vocab):
                self.assertLessEqual(abs(h[i][j] - h_gpu_mem[i][j]), 0.25)