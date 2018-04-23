from reikna.linalg import MatrixMul
import reikna.cluda as cluda
import pycuda.cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import pycuda.gpuarray
import numpy as np



increment = ElementwiseKernel(
        "float *X, float *Y",
        "Y[i] = 1 + X[i]",
        "increment")

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
    Y = pycuda.gpuarray.empty(X.shape, dtype=X.dtype)
    sigmoid = ElementwiseKernel(
        "double *Y, double *X",
        "Y[i] = 1.0 / (1.0 + exp (-X[i]) )",
        "sigmoid")
    sigmoid(Y, X)
    return Y


def softmax_gpu(X):
    Y = pycuda.gpuarray.empty(X.shape, dtype=X.dtype)
    exp_sum = ReductionKernel(np.float64, neutral="0.0",
                           reduce_expr="a + b", map_expr="exp(X[i])",
                           arguments="double *X", name="exp_sum")
    softmax = ElementwiseKernel(
        "double *Y, double *X, double s",
        "Y[i] = exp (X[i]) / s",
        "softmax")
    s = exp_sum(X).get()
    softmax(Y, X, s)
    return Y