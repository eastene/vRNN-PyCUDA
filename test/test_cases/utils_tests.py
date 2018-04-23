import unittest
import numpy as np
import pycuda.gpuarray
from src.utils.cuda import modified_gemm_gpu


class CudaTestCase(unittest.TestCase):

    def test_matmul_gpu(self):
        a = np.random.uniform(0, 10, (10, 5))
        b = np.random.uniform(0, 10, (5, 4))
        c = np.random.uniform(0, 10, (10, 4))

        A = pycuda.gpuarray.to_gpu(a)
        B = pycuda.gpuarray.to_gpu(b)
        C = pycuda.gpuarray.to_gpu(c)
        Y = modified_gemm_gpu(A, B, C)

        y = np.matmul(a, b) + c

        y_gpu = np.empty((10, 4))
        Y.get(y_gpu)

        self.assertListEqual(y.tolist(), y_gpu.tolist())