import numpy as np
from model.utils import sigmoid, tanh, softmax

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class Layer:

    def __init__(self, input_size, layer_size):
        # nvidia GPUs require single precision floats

        # layer weights
        self.Wf = np.random.uniform(0, 0, (layer_size, layer_size + input_size)).astype(np.float32)  # forget gate
        self.Wi = np.random.uniform(0, 0, (layer_size, layer_size + input_size)).astype(np.float32)  # update gate
        self.Wc = np.random.uniform(0, 0, (layer_size, layer_size + input_size)).astype(np.float32)  # tanh gate
        self.Wo = np.random.uniform(0, 0, (layer_size, layer_size + input_size)).astype(np.float32)  # output gate
        self.Wy = np.random.uniform(0, 0, (input_size, layer_size)).astype(np.float32)  # hidden layer to output gate

        # biases
        self.bf = np.random.uniform(0, 0, (layer_size, 1)).astype(np.float32)  # forget gate
        self.bi = np.random.uniform(0, 0, (layer_size, 1)).astype(np.float32)  # update gate
        self.bc = np.random.uniform(0, 0, (layer_size, 1)).astype(np.float32)  # tanh gate
        self.bo = np.random.uniform(0, 0, (layer_size, 1)).astype(np.float32)  # output gate
        self.by = np.random.uniform(0, 0, (input_size, 1)).astype(np.float32)  # hidden layer to output gate

        # previous time states
        self.a = np.zeros((layer_size, 1)).astype(np.float32)
        self.c = np.zeros((layer_size, 1)).astype(np.float32)

        # alloc on and copy to GPU


    def forward_prop(self, x_t):
        # concatenate a and x for efficiency
        concat = np.concatenate((self.a, x_t), axis = 0)

        # compute internal gate values
        ft = sigmoid(self.Wf * concat + self.bf)
        it = sigmoid(self.Wi * concat + self.bi)
        cct = tanh(self.Wc * concat + self.bc)
        ot = sigmoid(self.Wo * concat + self.bo)

        # update next time states
        self.c = ft * self.c + it * cct
        self.a = ot * tanh(self.c)

        # compute predicted output
        yhat_t = softmax(self.a)

        return yhat_t

    def backward_prop(self):
        pass

    def serialize(self):
        pass

    def deserialize(self):
        pass

    def forward_prop_gpu(self):
        pass

    def backward_prop_gpu(self):
        pass