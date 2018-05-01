import numpy as np
import pycuda.autoinit
import pycuda.cumath
import pycuda.gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel


def square_gpu(X):
    square = ElementwiseKernel(
        "double *Y, double *X",
        "Y[i] = X[i] * X[i]",
        "square")
    Y = pycuda.gpuarray.empty_like(X)
    square(Y, X)
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