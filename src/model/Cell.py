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

        # used for sanity checking
        self.on_gpu = False

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
        # states
        self.ft_gpu = pycuda.gpuarray.to_gpu(self.ft)
        self.it_gpu = pycuda.gpuarray.to_gpu(self.it)
        self.cct_gpu = pycuda.gpuarray.to_gpu(self.ct)
        self.ot_gpu = pycuda.gpuarray.to_gpu(self.ot)

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

            ftx_gpu = thr.array((self.Wfx_gpu.shape[0], x_t_gpu.shape[1]), dtype=np.float64)
            itx_gpu = thr.array((self.Wix_gpu.shape[0], x_t_gpu.shape[1]), dtype=np.float64)
            ctx_gpu = thr.array((self.Wcx_gpu.shape[0], x_t_gpu.shape[1]), dtype=np.float64)
            otx_gpu = thr.array((self.Wox_gpu.shape[0], x_t_gpu.shape[1]), dtype=np.float64)

            mul = MatrixMul(self.Wfx_gpu, x_t_gpu, out_arr=ftx_gpu)
            mulx = mul.compile(thr)
            mulx(ftx_gpu, self.Wfx_gpu, x_t_gpu)
            mulx(itx_gpu, self.Wfx_gpu, x_t_gpu)
            mulx(ctx_gpu, self.Wfx_gpu, x_t_gpu)
            mulx(otx_gpu, self.Wfx_gpu, x_t_gpu)

            fth_gpu = thr.array((self.Wfx_gpu.shape[0], x_t_gpu.shape[1]), dtype=np.float64)
            ith_gpu = thr.array((self.Wix_gpu.shape[0], x_t_gpu.shape[1]), dtype=np.float64)
            cth_gpu = thr.array((self.Wcx_gpu.shape[0], x_t_gpu.shape[1]), dtype=np.float64)
            oth_gpu = thr.array((self.Wox_gpu.shape[0], x_t_gpu.shape[1]), dtype=np.float64)

            mul = MatrixMul(self.Wfh_gpu, h_prev_gpu, out_arr=fth_gpu)
            mulh = mul.compile(thr)
            mulh(fth_gpu, self.Wfh_gpu, h_prev_gpu)
            mulh(ith_gpu, self.Wih_gpu, h_prev_gpu)
            mulh(cth_gpu, self.Wch_gpu, h_prev_gpu)
            mulh(oth_gpu, self.Woh_gpu, h_prev_gpu)

            self.ft_gpu = sigmoid_gpu(ftx_gpu +fth_gpu + self.bf_gpu)
            self.it_gpu = sigmoid_gpu(itx_gpu + ith_gpu + self.bi_gpu)
            self.cct_gpu = tanh_gpu(ctx_gpu + cth_gpu + self.bc_gpu)
            self.ot_gpu = sigmoid_gpu(otx_gpu + oth_gpu + self.bo_gpu)

            # update next time states
            cf_gpu = thr.array((fth_gpu.shape[0], c_prev_gpu.shape[1]), dtype=np.float64)
            ci_gpu = thr.array((fth_gpu.shape[0], c_prev_gpu.shape[1]), dtype=np.float64)
            mul = MatrixMul(self.ft_gpu, c_prev_gpu, out_arr=cf_gpu)
            mulc = mul.compile(thr)
            mulc(cf_gpu, self.ft_gpu, c_prev_gpu)
            mulc(ci_gpu, self.it_gpu, self.cct_gpu)

            c_gpu = cf_gpu + ci_gpu

            ac_gpu = tanh_gpu(c_gpu)
            h_gpu = thr.array((self.ot_gpu.shape[0], ac_gpu.shape[1]), dtype=np.float64)
            mul = MatrixMul(self.ft_gpu, c_prev_gpu, out_arr=cf_gpu)
            mulo = mul.compile(thr)
            mulo(h_gpu, self.ot_gpu, ac_gpu)

            return h_gpu, c_gpu

        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))

    def backward_prop_gpu(self):
        if self.on_gpu:
            pass
        else:
            raise(GPUException("Cell {0} not found on GPU".format(self.cell_cords)))