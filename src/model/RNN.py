from src.model.Layer import Layer
from src.utils.activations import softmax

class RNN:

    def __init__(self, num_unroll, vocab_size, batch_size, num_layers):
        # set parameters
        self.num_unroll = num_unroll
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.layers = [Layer(num_unroll, vocab_size, batch_size, i) for i in range(num_layers)]

    def __repr__(self):
        num_layers = 1 + len(self.hidden_layers) + 1
        rep = "{0} layers: \n".format(num_layers)
        rep += "  Input layer of size {0}\n".format(self.input_layer)
        for layer in self.hidden_layers:
            rep += "  Hidden layer of size {0}\n".format(layer)
        rep += "  Output layer of size {0}\n".format(self.output_layer)
        return rep

    def train(self, batch_generator):
        for t in range(self.num_unroll):
            x_t, y_t = batch_generator.next()
            # forward prop
            a = self.layers[0].forward_prop(x_t)
            for layer in self.layers[1:]:
                a = layer.forward_prop(a)
            yhat_t = softmax(a)
            # backward prop

        self.backward_prop(yhat_t, X.__next__)

    def forward_prop(self, x_t):
        yhat_t = self.input_layer.forward_prop(x_t)
        for layer in self.hidden_layers:
            yhat_t = layer.forward_prop(yhat_t)
        yhat_t = self.output_layer.forward_prop(yhat_t)

        return yhat_t

    def backward_prop(self, yhat_t, y_t):
        #retrieve the dimensions
        
        #initialize gradients with right sizes
        yhat_t = self.input_layer.backward_prop()
        for layer in self.hidden_layers:
        #     = layer.backward_prop()
            pass
        

    def serialize(self):
        pass

    def deserialize(self):
        pass
