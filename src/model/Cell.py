import numpy as np
from src.utils.activations import sigmoid, tanh, softmax
from src.utils.cuda import modified_gemm_gpu

from src.model.GPUException import GPUException

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray


class Cell:

    def __init__(self, vocab_size, batch_size, cell_cords):
        # initialize cell parameters
        self.input_size = vocab_size
        self.batch_size = batch_size
        self.cell_cords = cell_cords # coordinates of cell in RNN (layer, time)

        # layer weights
        self.Wf = np.random.uniform(-0.1, 0.1, (batch_size, batch_size + vocab_size))  # forget gate
        self.Wi = np.random.uniform(-0.1, 0.1, (batch_size, batch_size + vocab_size))  # update gate
        self.Wc = np.random.uniform(-0.1, 0.1, (batch_size, batch_size + vocab_size))  # tanh gate
        self.Wo = np.random.uniform(-0.1, 0.1, (batch_size, batch_size + vocab_size))  # output gate
        self.Wy = np.random.uniform(-0.1, 0.1, (vocab_size, batch_size))  # hidden layer to output gate

        # biases
        self.bf = np.zeros((batch_size, 1))  # forget gate
        self.bi = np.zeros((batch_size, 1))  # update gate
        self.bc = np.zeros((batch_size, 1))  # tanh gate
        self.bo = np.zeros((batch_size, 1))  # output gate
        self.by = np.zeros((vocab_size, 1))  # hidden layer to output gate

        # used for sanity checking
        self.on_gpu = False

    def forward_prop(self, c_prev, y_hat_prev, x_t):
        # concatenate a and x for efficiency
        concat = np.concatenate((y_hat_prev, x_t), axis = 0)

        # compute internal gate values
        ft = sigmoid(np.matmul(self.Wf, concat) + self.bf)
        it = sigmoid(np.matmul(self.Wi, concat)+ self.bi)
        cct = tanh(np.matmul(self.Wc, concat) + self.bc)
        ot = sigmoid(np.matmul(self.Wo, concat) + self.bo)

        # update next time states
        c = ft * c_prev + it * cct
        a = ot * tanh(c)

        #store values needed for backward propagation in cache       
        return a, c

    def backward_prop(self):
        
        """
        dx = np.zeros((self.input_size,m,self.hidden_layers))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros(da0.shape)
        dc_prevt = np.zeros(da0.shape)
        dWf = np.zeros((n_a, n_a + input_size))
        dWi = np.zeros(dWf.shape)
        dWc = np.zeros(dWf.shape)
        dWo = np.zeros(dWf.shape)
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros(dbf.shape)
        dbc = np.zeros(dbf.shape)
        dbo = np.zeros(dbf.shape)
        """
        #compute derivates of the gates
        dot = self.a * tanh(self.c) * ot * (1 - ot)
        dcct = (self.c * it + ot * (1 - np.square(tanh(self.c))) * it * self.a) * (1 - np.square(cct))
        dit = (self.c * cct + ot * (1 - np.square(tanh(self.c))) * cct * self.a) * it * (1 - it)
        dft = (self.c * self.c + ot *(1 - np.square(tanh(self.c))) * self.c * self.a) * ft * (1 - ft)

        # compute parameters  derivatives
        dWf = np.dot(dft, concat.T)
        dWi = np.dot(dit, concat.T)
        dWc = np.dot(dcct, concat.T)
        dWo = np.dot(dot, concat.T)
        dbf = np.sum(dft, axis=1 ,keepdims = True)
        dbi = np.sum(dit, axis=1, keepdims = True)
        dbc = np.sum(dcct, axis=1,  keepdims = True)
        dbo = np.sum(dot, axis=1, keepdims = True)
        
        #compute derivatives with respect to previous hidden state,memory and input.
        da_prev = np.dot(Wf[:, :layer_size].T,df)+np.dot(Wi[:, :layer_size],dit)+ np.dot(Wc[:, :layer_size].T, dcct) + np.dot(Wo[:, :layer_size].T, dot)
        dc_prev = self.c * ft + ot * (1 - np.square(tanh(c_next))) * ft * self.a
        dxt = np.dot(Wf[:, layer_size:].T, dft) + np.dot(Wi[:, layer_size:].T, dit) + np.dot(Wc[:, layer_size:].T, dcct) + np.dot(Wo[:, layer_size:].T, dot)

    def serialize(self):
        pass

    def deserialize(self):
        pass

    def cell_to_gpu(self):
        # alloc on and copy to GPU using GPUArrays doc: https://documen.tician.de/pycuda/array.html
        # weights
        self.Wf_gpu = pycuda.gpuarray.to_gpu(self.Wf)
        self.Wi_gpu = pycuda.gpuarray.to_gpu(self.Wi)
        self.Wc_gpu = pycuda.gpuarray.to_gpu(self.Wc)
        self.Wo_gpu = pycuda.gpuarray.to_gpu(self.Wo)
        self.Wy_gpu = pycuda.gpuarray.to_gpu(self.Wy)
        # biases
        self.bf_gpu = pycuda.gpuarray.to_gpu(self.bf)
        self.bi_gpu = pycuda.gpuarray.to_gpu(self.bi)
        self.bc_gpu = pycuda.gpuarray.to_gpu(self.bc)
        self.bo_gpu = pycuda.gpuarray.to_gpu(self.bo)
        self.by_gpu = pycuda.gpuarray.to_gpu(self.by)

        self.on_gpu = True

    def cell_from_gpu(self):
        if self.on_gpu:
            # weights
            self.Wf = self.Wf_gpu.get()
            self.Wi = self.Wi_gpu.get()
            self.Wc = self.Wc_gpu.get()
            self.Wo = self.Wo_gpu.get()
            self.Wy = self.Wy_gpu.get()
            # biases
            self.bf = self.bf_gpu.get()
            self.bi = self.bi_gpu.get()
            self.bc = self.bc_gpu.get()
            self.bo = self.bo_gpu.get()
            self.by = self.by_gpu.get()

            # remove
            del self.Wf_gpu
            del self.Wi_gpu
            del self.Wc_gpu
            del self.Wo_gpu
            del self.Wy_gpu
            del self.bf_gpu
            del self.bi_gpu
            del self.bc_gpu
            del self.bo_gpu
            del self.by_gpu

            self.on_gpu = False
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))

    def forward_prop_gpu(self, c_prev_gpu, y_hat_prev_gpu, x_t_gpu):
        if self.on_gpu:
            concat = None
            self.ft_gpu = modified_gemm_gpu(self.Wf, concat, self.bf)
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))

    def backward_prop_gpu(self):
        if self.on_gpu:
            pass
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))