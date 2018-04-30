import unittest
import numpy as np
import pycuda.curandom
import pycuda.gpuarray
from pycuda.tools import mark_cuda_test
from src.utils.cuda import *
from src.utils.activations import *

class CudaTestCase(unittest.TestCase):

    @mark_cuda_test
    def test_matmul_gpu(self):
        api = cluda.cuda_api()
        thr = api.Thread.create()

        a = np.random.uniform(0, 10, (10, 5))
        b = np.random.uniform(0, 10, (5, 4))

        A = pycuda.gpuarray.to_gpu(a)
        B = pycuda.gpuarray.to_gpu(b)
        Y = matmul_gpu(A, B, thr)

        y = np.matmul(a, b)

        y_gpu = Y.get()

        print(y)
        print(y_gpu)

        self.assertListEqual(y_gpu.tolist(), y.tolist())


    def test_add_bias(self):
        a = np.random.uniform(0, 10, (10, 5))
        b = np.ones((10, 1))

        x = a + b
        a_gpu = pycuda.gpuarray.to_gpu(a)
        b_gpu = pycuda.gpuarray.to_gpu(b)
        x_gpu = add_bias(a_gpu, b_gpu).get()

        self.assertListEqual(x_gpu.tolist(), x.tolist())


    @mark_cuda_test
    def test_tanh_gpu(self):
        shape = (4, 4)
        a = np.random.uniform(-100, 100, shape)
        A = pycuda.gpuarray.to_gpu(a)
        x_gpu = tanh_gpu(A).get()
        x = tanh(a)

        for i in range(shape[0]):
            for j in range(shape[1]):
                self.assertLessEqual(abs(x_gpu[i][j] - x[i][j]), 1e-5)

    @mark_cuda_test
    def test_sigmoid_gpu(self):
        shape = (4, 4)
        a = np.random.uniform(-100, 100, shape)
        A = pycuda.gpuarray.to_gpu(a)
        x_gpu = sigmoid_gpu(A).get()
        x = sigmoid(a)

        for i in range(shape[0]):
            for j in range(shape[1]):
                self.assertLessEqual(abs(x_gpu[i][j] - x[i][j]), 1e-5)

    @mark_cuda_test
    def test_softmax_gpu(self):
        shape = (4, 4)
        a = np.random.uniform(-100, 100, shape)
        A = pycuda.gpuarray.to_gpu(a)
        x_gpu = softmax_gpu(A).get()
        x = softmax(a)

        print(x)
        print(x_gpu)

        for i in range(shape[0]):
            for j in range(shape[1]):
                self.assertLessEqual(abs(x_gpu[i][j] - x[i][j]), 1e-5)
