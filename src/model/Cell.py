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
        self.Wfx = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, vocab_size))  # forget gate
        self.Wix = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, vocab_size))  # update gate
        self.Wcx = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, vocab_size))  # tanh gate
        self.Wox = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, vocab_size))  # output gate

        self.Wfh = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, self.hidden_state_size))  # forget gate
        self.Wih = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, self.hidden_state_size))  # update gate
        self.Wch = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, self.hidden_state_size))  # tanh gate
        self.Woh = np.random.uniform(-0.1, 0.1, (self.hidden_state_size, self.hidden_state_size))  # output gate

        # biases
        self.bf = np.zeros((self.hidden_state_size, batch_size))  # forget gate
        self.bi = np.zeros((self.hidden_state_size, batch_size))  # update gate
        self.bc = np.zeros((self.hidden_state_size, batch_size))  # tanh gate
        self.bo = np.zeros((self.hidden_state_size, batch_size))  # output gate

        # internal values
        self.ft = np.zeros((self.hidden_state_size, batch_size))
        self.it = np.zeros((self.hidden_state_size, batch_size))
        self.cct = np.zeros((self.hidden_state_size, batch_size))
        self.ot = np.zeros((self.hidden_state_size, batch_size))

        # previous states
        self.h_prev = np.zeros(self.hidden_state_size, self.batch_size)
        self.c_prev = np.zeros(self.hidden_state_size, self.batch_size)
        self.x_t = np.zeros(self.hidden_state_size, self.batch_size)

        # current states
        self.h = np.zeros(self.hidden_state_size, self.batch_size)
        self.c = np.zeros(self.hidden_state_size, self.batch_size)

        # used for sanity checking
        self.on_gpu = False

        # GPU array handles
        self.Wfx_gpu = None
        self.Wix_gpu = None
        self.Wcx_gpu = None
        self.Wox_gpu = None
        self.Wfh_gpu = None
        self.Wih_gpu = None
        self.Wch_gpu = None
        self.Woh_gpu = None
        # biases
        self.bf_gpu = None
        self.bi_gpu = None
        self.bc_gpu = None
        self.bo_gpu = None
        # states
        self.ft_gpu = None
        self.it_gpu = None
        self.cct_gpu = None
        self.ot_gpu = None

    def forward_prop(self, c_prev, h_prev, x_t):
        # normally would concatenate a and x for efficiency, but for demonstration it is easier to keep separate

        # compute internal gate values
        self.ft = sigmoid(np.matmul(self.Wfx, x_t) + np.matmul(self.Wfh, h_prev) + self.bf)
        self.it = sigmoid(np.matmul(self.Wix, x_t) + np.matmul(self.Wih, h_prev) + self.bi)
        self.cct = tanh(np.matmul(self.Wcx, x_t) + np.matmul(self.Wch, h_prev) + self.bc)
        self.ot = sigmoid(np.matmul(self.Wox, x_t) + np.matmul(self.Woh, h_prev) + self.bo)

        # update next time states
        c = self.ft * c_prev + self.it * self.cct
        h = self.ot * tanh(c)


        return h, c

    def backward_prop(self, da_next, dc_next):

        dot = da_next * tanh(self.c) * self.ot * (1 - self.ot)
        dcct = dc_next * self.it + self.ot * (1 - tanh(self.c)**2) * self.it * da_next
        dit = None
        dft = None

        # Code equations (7) to (10) (≈4 lines)
        dit = None
        dft = None
        dot = None
        dcct = None

        # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
        dWf = None
        dWi = None
        dWc = None
        dWo = None
        dbf = None
        dbi = None
        dbc = None
        dbo = None

        # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
        da_prev = None
        dc_prev = None
        dxt = None

    def serialize(self):
        pass

    def deserialize(self):
        pass

    def cell_to_gpu(self):
        if not self.on_gpu:
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
            # states
            self.ft_gpu = pycuda.gpuarray.to_gpu(self.ft)
            self.it_gpu = pycuda.gpuarray.to_gpu(self.it)
            self.cct_gpu = pycuda.gpuarray.to_gpu(self.cct)
            self.ot_gpu = pycuda.gpuarray.to_gpu(self.ot)

            self.on_gpu = True
        else:
            raise(GPUException("Cell {0} already on GPU".format(self.cell_cords)))

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

            # state
            self.ft = self.ft_gpu.get()
            self.it = self.it_gpu.get()
            self.cct = self.cct_gpu.get()
            self.ot = self.ot_gpu.get()

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
            del self.ft_gpu
            del self.it_gpu
            del self.cct_gpu
            del self.ot_gpu

            self.on_gpu = False
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))

    def forward_prop_gpu(self, c_prev_gpu, h_prev_gpu, x_t_gpu):
        if self.on_gpu:
            api = cluda.cuda_api()
            thr = api.Thread.create()

            self.ft_gpu = sigmoid_gpu(
                matmul_gpu(self.Wfx_gpu, x_t_gpu, ) + matmul_gpu(self.Wfh_gpu, h_prev_gpu, ) + self.bf_gpu)
            self.it_gpu = sigmoid_gpu(
                matmul_gpu(self.Wix_gpu, x_t_gpu, ) + matmul_gpu(self.Wih_gpu, h_prev_gpu, ) + self.bi_gpu)
            self.cct_gpu = tanh_gpu(
                matmul_gpu(self.Wcx_gpu, x_t_gpu, ) + matmul_gpu(self.Wch_gpu, h_prev_gpu, ) + self.bc_gpu)
            self.ot_gpu = sigmoid_gpu(
                matmul_gpu(self.Wox_gpu, x_t_gpu, ) + matmul_gpu(self.Woh_gpu, h_prev_gpu, ) + self.bo_gpu)

            c_gpu = matmul_gpu(self.ft_gpu, c_prev_gpu, ) + matmul_gpu(self.it_gpu, self.cct_gpu, )
            h_gpu = matmul_gpu(self.ot_gpu, tanh_gpu(c_gpu), )

            return h_gpu, c_gpu

        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))

    def backward_prop_gpu(self):
        if self.on_gpu:
            pass
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))