from src.model.Layer import Layer


class RNN:

    def __init__(self, vocab_size, hidden_layers):
        # set parameters
        self.vocab_size = vocab_size
        self.hidden_layers = [Layer(previous_layer, layer) for previous_layer, layer in
                              zip([vocab_size] + hidden_layers, hidden_layers)]
        self.input_layer = Layer(vocab_size, vocab_size)
        self.output_layer = Layer(hidden_layers[-1], vocab_size)

    def __repr__(self):
        num_layers = 1 + len(self.hidden_layers) + 1
        rep = "{0} layers: \n".format(num_layers)
        rep += "  Input layer of size {0}\n".format(self.input_layer)
        for layer in self.hidden_layers:
            rep += "  Hidden layer of size {0}\n".format(layer)
        rep += "  Output layer of size {0}\n".format(self.output_layer)
        return rep

    def train(self, X):
        for x in X:
            yhat_t = self.forward_prop(x)
            self.backward_prop(yhat_t, X.__next__)

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
