import numpy as np
from src.utils.activations import sigmoid, tanh, softmax
from src.utils.cuda import *

from src.model.GPUException import GPUException

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray


class Cell:

    def __init__(self, vocab_size, batch_size, cell_cords):
        # initialize cell parameters
        self.input_size = vocab_size
        self.hidden_state_size = vocab_size  # can be changed in the future
        self.batch_size = batch_size
        self.cell_cords = cell_cords # coordinates of cell in RNN (layer, time)

        # layer weights
        self.Wfx = np.random.uniform(-0.1, 0.1, (batch_size, vocab_size))  # forget gate
        self.Wix = np.random.uniform(-0.1, 0.1, (batch_size, vocab_size))  # update gate
        self.Wcx = np.random.uniform(-0.1, 0.1, (batch_size, vocab_size))  # tanh gate
        self.Wox = np.random.uniform(-0.1, 0.1, (batch_size, vocab_size))  # output gate

        self.Wfh = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, self.hidden_state_size))  # forget gate
        self.Wih = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, self.hidden_state_size))  # update gate
        self.Wch = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, self.hidden_state_size))  # tanh gate
        self.Woh = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, self.hidden_state_size))  # output gate

        # biases
        self.bf = np.zeros((batch_size, 1))  # forget gate
        self.bi = np.zeros((batch_size, 1))  # update gate
        self.bc = np.zeros((batch_size, 1))  # tanh gate
        self.bo = np.zeros((batch_size, 1))  # output gate

        # used for sanity checking
        self.on_gpu = False

    def forward_prop(self, c_prev, h_prev, x_t):
        # normally would concatenate a and x for efficiency, but for demonstration it is easier to keep separate

        # compute internal gate values
        ft = sigmoid(np.matmul(x_t, self.Wfx) + np.matmul(h_prev, self.Wfh) + self.bf)
        it = sigmoid(np.matmul(x_t, self.Wix) + np.matmul(h_prev, self.Wih) + self.bi)
        cct = tanh(np.matmul(x_t, self.Wcx) + np.matmul(h_prev, self.Wch) + self.bc)
        ot = sigmoid(np.matmul(x_t, self.Wox) + np.matmul(h_prev, self.Woh) + self.bo)

        # update next time states
        c = ft * c_prev + it * cct
        h = ot * tanh(c)

        return h, c

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
        self.Wfx_gpu = pycuda.gpuarray.to_gpu(self.Wfx)
        self.Wix_gpu = pycuda.gpuarray.to_gpu(self.Wix)
        self.Wcx_gpu = pycuda.gpuarray.to_gpu(self.Wcx)
        self.Wox_gpu = pycuda.gpuarray.to_gpu(self.Wox)
        self.Wfh_gpu = pycuda.gpuarray.to_gpu(self.Wfh)
        self.Wih_gpu = pycuda.gpuarray.to_gpu(self.Wih)
        self.Wch_gpu = pycuda.gpuarray.to_gpu(self.Wch)
        self.Woh_gpu = pycuda.gpuarray.to_gpu(self.Woh)
        # biases
        self.bf_gpu = pycuda.gpuarray.to_gpu(self.bf)
        self.bi_gpu = pycuda.gpuarray.to_gpu(self.bi)
        self.bc_gpu = pycuda.gpuarray.to_gpu(self.bc)
        self.bo_gpu = pycuda.gpuarray.to_gpu(self.bo)

        self.on_gpu = True

    def cell_from_gpu(self):
        if self.on_gpu:
            # weights
            self.Wfx = self.Wfx_gpu.get()
            self.Wix = self.Wix_gpu.get()
            self.Wcx = self.Wcx_gpu.get()
            self.Wox = self.Wox_gpu.get()
            self.Wfh = self.Wfh_gpu.get()
            self.Wih = self.Wih_gpu.get()
            self.Wch = self.Wch_gpu.get()
            self.Woh = self.Woh_gpu.get()

            # biases
            self.bf = self.bf_gpu.get()
            self.bi = self.bi_gpu.get()
            self.bc = self.bc_gpu.get()
            self.bo = self.bo_gpu.get()

            # remove
            del self.Wfx_gpu
            del self.Wix_gpu
            del self.Wcx_gpu
            del self.Wox_gpu
            del self.Wfh_gpu
            del self.Wih_gpu
            del self.Wch_gpu
            del self.Woh_gpu
            del self.bf_gpu
            del self.bi_gpu
            del self.bc_gpu
            del self.bo_gpu

            self.on_gpu = False
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))

    def forward_prop_gpu(self, c_prev_gpu, h_prev_gpu, x_t_gpu):
        if self.on_gpu:
            ft_gpu = sigmoid_gpu(matmul_gpu(x_t_gpu, self.Wfx_gpu) + matmul_gpu(h_prev_gpu, self.Wfh_gpu) + self.bf_gpu)
            it_gpu = sigmoid_gpu(matmul_gpu(x_t_gpu, self.Wix_gpu) + matmul_gpu(h_prev_gpu, self.Wih_gpu) + self.bi_gpu)
            cct_gpu = tanh_gpu(matmul_gpu(x_t_gpu, self.Wcx_gpu) + matmul_gpu(h_prev_gpu, self.Wch_gpu) + self.bc_gpu)
            ot_gpu = sigmoid_gpu(matmul_gpu(x_t_gpu, self.Wox_gpu) + matmul_gpu(h_prev_gpu, self.Woh_gpu) + self.bo_gpu)

            # update next time states
            c_gpu = matmul_gpu(ft_gpu, c_prev_gpu) + matmul_gpu(it_gpu, cct_gpu)
            h_gpu = ot_gpu * tanh_gpu(c_gpu)

            return h_gpu, c_gpu
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))

    def backward_prop_gpu(self):
        if self.on_gpu:
            pass
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))