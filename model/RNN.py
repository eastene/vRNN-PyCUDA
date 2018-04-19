from model.Layer import Layer
import numpy as np

class RNN:

    def __init__(self, vocab_size, hidden_layers):
        # set parameters
        self.vocab_size = vocab_size
        self.hidden_layers = [Layer(previous_layer, layer) for previous_layer, layer in zip([vocab_size] + hidden_layers, hidden_layers)]
        self.input_layer = Layer(vocab_size, vocab_size)
        self.output_layer = Layer(hidden_layers.tail(), vocab_size)

    def train(self, X):
        T = len(X)
        for t in range(T - 1):
            yhat_t = self.forward_prop(X[t])
            self.backward_prop(yhat_t, X[t + 1])

    def forward_prop(self, x_t):
        yhat_t = self.input_layer.forward_prop(x_t)
        for layer in self.hidden_layers:
            yhat_t = layer.forward_prop(yhat_t)
        yhat_t = self.output_layer.forward_prop(yhat_t)

        return yhat_t

    def backward_prop(self, yhat_t, y_t):
        pass

    def serialize(self):
        pass

    def deserialize(self):
        pass
