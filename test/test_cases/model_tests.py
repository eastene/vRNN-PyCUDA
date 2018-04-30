import unittest
import numpy as np
from time import time
from src.model.LSTM import LSTM
from src.model.lstm_layer import *
import random

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

    def test_train(self):
        num_unroll = 10
        vocab_size = 1000
        batch_size = 50
        num_layers = 3
        learning_rate = 0.05

        lstm = LSTM(num_unroll, vocab_size, batch_size, num_layers, learning_rate)

        parameters = lstm.allocate_parameters()
        caches_cache = []

        X = np.zeros((vocab_size, batch_size, num_unroll + 1))
        for i in range(num_unroll + 1):
            for j in range(batch_size):
                X[random.randrange(0, vocab_size), j, i] = 1

        a0 = np.zeros((vocab_size, batch_size))
        start = time()
        a, y, c, caches = lstm_forward(X[:, :, :num_unroll], a0, parameters[0])
        caches_cache.append(caches)
        for layer in range(1, num_layers):
            a, y, c, caches = lstm_forward(y, a, parameters[layer])
            caches_cache.append(caches)

        loss = X[:, :, 1:] - y

        gradients = lstm_backward(loss, caches_cache[len(caches_cache) - 1])
        update_weights(parameters[num_layers - 1], gradients, learning_rate)
        for layer in reversed(range(num_layers - 1)):
            gradients = lstm_backward(gradients['dx'], caches_cache[layer])
            update_weights(parameters[layer], gradients, learning_rate)
        end = time()

        print(end - start)

class LstmLayerTestCase(unittest.TestCase):

    @mark_cuda_test
    def test_add(self):
        np.random.seed(1)
        Wy = np.random.randn(2, 10)
        by = np.random.randn(2, 10)

        Wy_gpu = pycuda.gpuarray.to_gpu(Wy)
        by_gpu = pycuda.gpuarray.to_gpu(by)

        print(Wy + by)
        print((Wy_gpu + by_gpu).get())

    @mark_cuda_test
    def test_forward_prop(self):
        np.random.seed(1)
        x = np.random.randn(3, 10)
        a0 = np.random.randn(5, 10)
        Wf = np.random.randn(5, 5 + 3)
        bf = np.random.randn(5, 1)
        Wi = np.random.randn(5, 5 + 3)
        bi = np.random.randn(5, 1)
        Wo = np.random.randn(5, 5 + 3)
        bo = np.random.randn(5, 1)
        Wc = np.random.randn(5, 5 + 3)
        bc = np.random.randn(5, 1)
        Wy = np.random.randn(2, 5)
        by = np.random.randn(2, 1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc,
                      "by": by}

        Wf_gpu = pycuda.gpuarray.to_gpu(Wf)
        Wi_gpu = pycuda.gpuarray.to_gpu(Wi)
        Wo_gpu = pycuda.gpuarray.to_gpu(Wo)
        Wc_gpu = pycuda.gpuarray.to_gpu(Wc)
        Wy_gpu = pycuda.gpuarray.to_gpu(Wy)
        bf_gpu = pycuda.gpuarray.to_gpu(bf)
        bi_gpu = pycuda.gpuarray.to_gpu(bi)
        bo_gpu = pycuda.gpuarray.to_gpu(bo)
        bc_gpu = pycuda.gpuarray.to_gpu(bc)
        by_gpu = pycuda.gpuarray.to_gpu(by)

        parameters_gpu = {"Wf": Wf_gpu, "Wi": Wi_gpu, "Wo": Wo_gpu, "Wc": Wc_gpu, "Wy": Wy_gpu, "bf": bf_gpu, "bi": bi_gpu, "bo": bo_gpu, "bc": bc_gpu,
                      "by": by_gpu}

        a, y, c, caches = lstm_cell_forward(x, a0, a0, parameters)
        print("CPU DONE")
        a_gpu, y_gpu, c_gpu, caches = lstm_cell_forward_gpu(x, a0, a0, parameters_gpu)
        print("GPU DONE")

        print(a)
        print(a_gpu.get())

    @mark_cuda_test
    def test_back_prop(self):
        np.random.seed(1)
        x = np.random.randn(3, 10)
        a0 = np.random.randn(5, 10)
        Wf = np.random.randn(5, 5 + 3)
        bf = np.random.randn(5, 1)
        Wi = np.random.randn(5, 5 + 3)
        bi = np.random.randn(5, 1)
        Wo = np.random.randn(5, 5 + 3)
        bo = np.random.randn(5, 1)
        Wc = np.random.randn(5, 5 + 3)
        bc = np.random.randn(5, 1)
        Wy = np.random.randn(2, 5)
        by = np.random.randn(2, 1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc,
                      "by": by}

        Wf_gpu = pycuda.gpuarray.to_gpu(Wf)
        Wi_gpu = pycuda.gpuarray.to_gpu(Wi)
        Wo_gpu = pycuda.gpuarray.to_gpu(Wo)
        Wc_gpu = pycuda.gpuarray.to_gpu(Wc)
        Wy_gpu = pycuda.gpuarray.to_gpu(Wy)
        bf_gpu = pycuda.gpuarray.to_gpu(bf)
        bi_gpu = pycuda.gpuarray.to_gpu(bi)
        bo_gpu = pycuda.gpuarray.to_gpu(bo)
        bc_gpu = pycuda.gpuarray.to_gpu(bc)
        by_gpu = pycuda.gpuarray.to_gpu(by)

        parameters_gpu = {"Wf": Wf_gpu, "Wi": Wi_gpu, "Wo": Wo_gpu, "Wc": Wc_gpu, "Wy": Wy_gpu, "bf": bf_gpu,
                          "bi": bi_gpu, "bo": bo_gpu, "bc": bc_gpu,
                          "by": by_gpu}

        a, y, c, caches = lstm_cell_forward(x, a0, a0, parameters)
        da_next = np.random.randn(5, 10)
        dc_next = np.random.randn(5, 10)
        gradients = lstm_cell_backward(da_next, dc_next, caches)
        print("CPU DONE")
        a_gpu, y_gpu, c_gpu, caches_gpu = lstm_cell_forward_gpu(x, a0, a0, parameters_gpu)
        da_next_gpu = pycuda.gpuarray.to_gpu(da_next)
        dc_next_gpu = pycuda.gpuarray.to_gpu(dc_next)
        gradients_gpu = lstm_cell_backward_gpu(da_next_gpu, dc_next_gpu, caches_gpu)
        print("GPU DONE")

        print(gradients['dbo'])
        print(gradients_gpu['dbo'].get())


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