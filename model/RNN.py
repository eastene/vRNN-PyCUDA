from model.Layer import Layer
import numpy as np

class RNN:

    def __init__(self, vocab_size, hidden_layers):
        # set parameters
        self.vocab_size = vocab_size
        self.hidden_layers = [Layer(vocab_size, layer) for layer in hidden_layers]
        self.input_layer = Layer(vocab_size, vocab_size)
        self.output_layer = Layer(vocab_size, vocab_size)

    def train(self, X):
        T = len(X)
        for t in range(T):
            yhat_t = self.forward_prop(X[t])

    def forward_prop(self, x_t):
        yhat_t = self.input_layer.forward_prop(x_t)
        for layer in self.hidden_layers:
            yhat_t = layer.forward_prop(yhat_t)
        yhat_t = self.output_layer.forward_prop(yhat_t)

        return yhat_t


    def backward_prop(self):
        pass

    def serialize(self):
        pass

    def deserialize(self):
        pass
