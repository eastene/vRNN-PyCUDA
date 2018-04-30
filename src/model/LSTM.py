import numpy as np
from src.model.lstm_layer import *
from src.utils.activations import softmax
from src.preprocess.VocabCoder import VocabCoder
from src.preprocess.BatchGenerator import BatchGenerator

class RNN:

    def __init__(self, num_unroll, vocab_size, batch_size, num_layers):
        # set parameters
        self.num_unroll = num_unroll
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.layer_size = [self.vocab_size for i in range(num_layers)]  # can be modified for different size layers

        self.layer_caches = []

    def train(self, vocab, text):
        coder = VocabCoder(vocab)
        batch_generator = BatchGenerator(text, self.batch_size, self.vocab_size, coder)
        parameters = self.allocate_parameters()
        for X in batch_generator:
            # forward prop
            a0 = np.zeros((self.layer_size[0], self.batch_size))
            a, y, c, caches = lstm_forward(X, a0, parameters[0])
            for layer in range(1, self.num_layers):
                a, y, c, caches = lstm_forward(y, a, parameters[layer])
            yhat_t = softmax(h)
            # backward prop

    def train_gpu(self, vocab, text):
        coder = VocabCoder(vocab)
        batch_generator = BatchGenerator(text, self.batch_size, self.vocab_size, coder)
        parameters = self.allocate_parameters()
        for X in batch_generator:
            # forward prop
            a0 = np.zeros((self.layer_size[0], self.batch_size))
            a, y, c, caches = lstm_forward_gpu(X, a0, parameters[0])
            for layer in range(1, self.num_layers):
                a, y, c, caches = lstm_forward_gpu(y, a, parameters[layer])
            yhat_t = softmax(h)
            # backward prop

    def allocate_parameters(self):
        parameters = []
        for i in range(self.num_layers):
            parameters_layer = []
            for j in range(self.num_unroll):
                Wf = np.random.randn(self.layer_size[i], self.layer_size[i] + self.batch_size)
                bf = np.random.randn(self.layer_size[i], 1)
                Wi = np.random.randn(self.layer_size[i], self.layer_size[i] + self.batch_size)
                bi = np.random.randn(self.layer_size[i], 1)
                Wo = np.random.randn(self.layer_size[i], self.layer_size[i] + self.batch_size)
                bo = np.random.randn(self.layer_size[i], 1)
                Wc = np.random.randn(self.layer_size[i], self.layer_size[i] + self.batch_size)
                bc = np.random.randn(self.layer_size[i], 1)
                Wy = np.random.randn(self.layer_size[i], self.layer_size[i])
                by = np.random.randn(self.layer_size[i], 1)
                parameters_cell = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo,
                                   "bc": bc, "by": by}
                parameters_layer.append(parameters_cell)
            parameters.append(parameters_layer)
        return parameters

    """
    def serialize(self):
        pass

    def deserialize(self):
        pass
    """
