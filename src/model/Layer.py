import numpy as np
from src.utils.activations import sigmoid, tanh, softmax
from src.utils.cuda import modified_gemm_gpu

import pycuda.autoinit
import pycuda.gpuarray


class Layer:

    def __init__(self, input_size, layer_size):
        self.input_size = input_size
        self.layer_size = layer_size
        # nvidia GPUs require single precision floats

        # layer weights
        self.Wf = np.random.uniform(0, 0, (layer_size, layer_size + input_size))  # forget gate
        self.Wi = np.random.uniform(0, 0, (layer_size, layer_size + input_size))  # update gate
        self.Wc = np.random.uniform(0, 0, (layer_size, layer_size + input_size))  # tanh gate
        self.Wo = np.random.uniform(0, 0, (layer_size, layer_size + input_size))  # output gate
        self.Wy = np.random.uniform(0, 0, (input_size, layer_size))  # hidden layer to output gate

        # biases
        self.bf = np.random.uniform(0, 0, (layer_size, 1))  # forget gate
        self.bi = np.random.uniform(0, 0, (layer_size, 1))  # update gate
        self.bc = np.random.uniform(0, 0, (layer_size, 1))  # tanh gate
        self.bo = np.random.uniform(0, 0, (layer_size, 1))  # output gate
        self.by = np.random.uniform(0, 0, (input_size, 1))  # hidden layer to output gate

        # previous time states
        self.a = np.zeros((layer_size, 1))
        self.c = np.zeros((layer_size, 1))

    def __repr__(self):
        return "{0} to {1}".format(self.input_size, self.layer_size)

    def forward_prop(self, x_t):
        # concatenate a and x for efficiency
        concat = np.concatenate((self.a, x_t), axis = 0)

        # compute internal gate values
        ft = sigmoid(np.matmul(self.Wf, concat) + self.bf)
        it = sigmoid(np.matmul(self.Wi, concat)+ self.bi)
        cct = tanh(np.matmul(self.Wc, concat) + self.bc)
        ot = sigmoid(np.matmul(self.Wo, concat) + self.bo)

        # update next time states
        self.c = ft * self.c + it * cct
        self.a = ot * tanh(self.c)

        # compute predicted output
        yhat_t = softmax(self.a)

        return yhat_t.astype(np.int64)

    def backward_prop(self):
        pass

    def serialize(self):
        pass

    def deserialize(self):
        pass

    def layer_to_gpu(self):
        # alloc on and copy to GPU using GPUArrays doc: https://documen.tician.de/pycuda/array.html
        # weights
        self.Wf_gpu = pycuda.gpuarray.to_gpu(self.Wf)
        Wi_gpu = pycuda.gpuarray.to_gpu(self.Wi)
        Wc_gpu = pycuda.gpuarray.to_gpu(self.Wc)
        Wo_gpu = pycuda.gpuarray.to_gpu(self.Wo)
        Wy_gpu = pycuda.gpuarray.to_gpu(self.Wy)
        # biases
        self.bf_gpu = pycuda.gpuarray.to_gpu(self.bf)
        bi_gpu = pycuda.gpuarray.to_gpu(self.bi)
        bc_gpu = pycuda.gpuarray.to_gpu(self.bc)
        bo_gpu = pycuda.gpuarray.to_gpu(self.bo)
        by_gpu = pycuda.gpuarray.to_gpu(self.by)
        # time states
        c_gpu = pycuda.gpuarray.to_gpu(self.c)
        self.a_gpu = pycuda.gpuarray.to_gpu(self.a)
        # outputs
        self.ft_gpu = pycuda.gpuarray.zeros((self.layer_size, 1), np.float64)
        it_gpu = pycuda.gpuarray.zeros((self.layer_size, 1), np.float64)
        cct_gpu = pycuda.gpuarray.zeros((self.layer_size, 1), np.float64)
        ot_gpu = pycuda.gpuarray.zeros((self.layer_size, 1), np.float64)
        yhat_gpu = pycuda.gpuarray.zeros((self.layer_size, 1), np.float64)

    def forward_prop_gpu(self, x_t_gpu):
        concat = self.a_gpu
        self.ft_gpu = modified_gemm_gpu(self.Wf, concat, self.bf)

    def backward_prop_gpu(self):
        pass