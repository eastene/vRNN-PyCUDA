import pycuda.autoinit
from reikna.linalg import MatrixMul
import reikna.cluda as cluda
import pycuda.cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import pycuda.gpuarray
import numpy as np


def modified_gemm_gpu(A, B, C):
    shape = (A.shape[0], B.shape[1])
    api = cluda.cuda_api()
    thr = api.Thread.create()
    res_arr = thr.array((shape[0], shape[1]), dtype=A.dtype)

    mul = MatrixMul(A, B, out_arr=res_arr)
    mulc = mul.compile(thr)
    mulc(res_arr, A, B)

    return res_arr + C


def tanh_gpu(X):
    return pycuda.cumath.tanh(X)


def sigmoid_gpu(X):
    sigmoid = ElementwiseKernel(
        "double *Y, double *X",
        "Y[i] = 1.0 / (1.0 + exp (-X[i]) )",
        "sigmoid")
    Y = pycuda.gpuarray.empty_like(X)
    sigmoid(Y, X)
    return Y


def softmax_gpu(X):
    exp_sum = ReductionKernel(np.float64, neutral="0.0",
            reduce_expr="a+b", map_expr="exp (x[i])",
            arguments="double *x")
    softmax = ElementwiseKernel(
        "double *Y, double *X, double s",
        "Y[i] = exp (X[i]) / s",
        "softmax")
    Y = pycuda.gpuarray.empty_like(X)
    s = exp_sum(X).get()
    softmax(Y, X, s)
    return Y