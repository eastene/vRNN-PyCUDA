import Layer
import numpy as np

class RNN:

    def __init__(self, vocab_size, hidden_layers, ):
        # set parameters
        self.vocab_size = vocab_size
        self.hidden_layers = hidden_layers

        # initialize weights
        self.U = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (hidden_layers, vocab_size))
        self.V = np.random.uniform(-np.sqrt(1./hidden_layers), np.sqrt(1./hidden_layers), (vocab_size, hidden_layers))
        self.W = np.random.uniform(-np.sqrt(1./hidden_layers), np.sqrt(1./hidden_layers), (hidden_layers, hidden_layers))

        pass

    def forward_prop(self):
        pass

    def backward_prop(self):
        pass