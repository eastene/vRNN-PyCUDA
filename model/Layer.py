import math
import numpy as np
from sklearn.utils.extmath import softmax

class Layer:

    def __init(self, vocab_size, layer_size, stride_size):

        # layer weights
        self.Wf = np.random.uniform(0, 0, (layer_size, layer_size + vocab_size))  # forget gate
        self.Wi = np.random.uniform(0, 0, (layer_size, layer_size + vocab_size))  # update gate
        self.Wc = np.random.uniform(0, 0, (layer_size, layer_size + vocab_size))  # tanh gate
        self.Wo = np.random.uniform(0, 0, (layer_size, layer_size + vocab_size))  # output gate
        self.Wy = np.random.uniform(0, 0, (vocab_size, layer_size))  # hidden layer to output gate

        # biases
        self.bf = np.random.uniform(0, 0, (layer_size, 1))  # forget gate
        self.bi = np.random.uniform(0, 0, (layer_size, 1))  # update gate
        self.bc = np.random.uniform(0, 0, (layer_size, 1))  # tanh gate
        self.bo = np.random.uniform(0, 0, (layer_size, 1))  # output gate
        self.by = np.random.uniform(0, 0, (vocab_size, 1))  # hidden layer to output gate

        # previous time states
        self.a = np.zeros((layer_size,))
        self.c = np.zeros((layer_size,))

    def forward_prop(self, x_t):
        # concatenate a and x for efficiency
        concat = np.concatenate((self.a, x_t), axis = 0)

        # compute internal gate values
        ft = math.sigmoid(self.Wf * concat + self.bf)
        it = math.sigmoid(self.Wi * concat + self.bi)
        cct = math.tanh(self.Wc * concat + self.bc)
        ot = math.sigmoid(self.Wo * concat + self.bo)

        # update next time states
        self.c = ft * self.c + it * cct
        self.a = ot * math.tanh(self.c)

        # compute predicted output
        yhat_t = softmax(self.a)

        return yhat_t