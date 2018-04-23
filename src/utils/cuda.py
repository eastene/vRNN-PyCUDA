from reikna.linalg import MatrixMul
import reikna.cluda as cluda
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