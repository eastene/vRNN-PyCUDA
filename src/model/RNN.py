from src.model.Layer import Layer
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

        self.layers = [Layer(num_unroll, vocab_size, batch_size, i) for i in range(num_layers)]

    def __repr__(self):
        num_layers = 1 + len(self.hidden_layers) + 1
        rep = "{0} layers: \n".format(num_layers)
        rep += "  Input layer of size {0}\n".format(self.input_layer)
        for layer in self.hidden_layers:
            rep += "  Hidden layer of size {0}\n".format(layer)
        rep += "  Output layer of size {0}\n".format(self.output_layer)
        return rep

    def train(self, vocab, text):
        coder = VocabCoder(vocab)
        stride = self.batch_size * self.num_unroll
        iterations = len(text) / stride
        batch_generator = BatchGenerator(text, self.batch_size, self.vocab_size, coder)
        for train, label in batch_generator:
            # forward prop
            h = self.layers[0].forward_prop(train)
            for layer in self.layers[1:]:
                h = layer.forward_prop(h)
            yhat_t = softmax(h)
            # backward prop

    def serialize(self):
        pass

    def deserialize(self):
        pass
