import pycuda.autoinit
from reikna.linalg import MatrixMul
import reikna.cluda as cluda
import pycuda.cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import pycuda.gpuarray
import numpy as np
from src.model.GPUException import GPUException

def matmul_gpu(A, B, thr):

    shape = (A.shape[0], B.shape[1])
    res_arr = thr.array((shape[0], shape[1]), dtype=A.dtype)

    mul = MatrixMul(A, B, out_arr=res_arr)
    mulc = mul.compile(thr)
    mulc(res_arr, A, B)

    return res_arr


def square_gpu(A, thr):

    shape = A.shape
    res_arr = thr.array((shape[0], shape[1]), dtype=A.dtype)

    mul = MatrixMul(A, A, out_arr=res_arr)
    mulc = mul.compile(thr)
    mulc(res_arr, A, A)

    return res_arr


def add_bias_gpu(X, b):
    len, m = X.shape
    add_bias = ElementwiseKernel(
        "double *Y, double *X, double *b, int len",
        "Y[i] = X[i] + b[i % len]",
        "add_bias")
    Y = pycuda.gpuarray.empty_like(X)
    add_bias(Y, X, b, len)
    return Y


def from_one_gpu(X):
    from_one = ElementwiseKernel(
        "double *Y, double *X",
        "Y[i] = 1.0 - X[i]",
        "from_one")
    Y = pycuda.gpuarray.empty_like(X)
    from_one(Y, X)
    return Y


def tanh_gpu(X):
    tanh_ = ElementwiseKernel(
        "double *Y, double *X",
        "Y[i] = (exp (X[i]) - exp (-X[i])) / (exp (X[i]) + exp (-X[i]))",
        "tanh_")
    Y = pycuda.gpuarray.empty_like(X)
    tanh_(Y, X)
    return Y


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