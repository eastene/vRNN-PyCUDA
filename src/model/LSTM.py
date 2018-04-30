from src.model.lstm_layer import *
from random import randrange
from src.preprocess.VocabCoder import VocabCoder
from src.preprocess.BatchGenerator import BatchGenerator
from random import randrange

from src.model.lstm_layer import *
from src.preprocess.BatchGenerator import BatchGenerator
from src.preprocess.VocabCoder import VocabCoder


class LSTM:

    def __init__(self, num_unroll, vocab_size, batch_size, num_layers, learning_rate):
        # set parameters
        self.num_unroll = num_unroll
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.layer_size = [self.vocab_size for i in range(num_layers + 1)]  # can be modified for different size layers
        self.layer_caches = []

        self.parameters = self.allocate_parameters()

    def train(self, vocab, text):
        coder = VocabCoder(vocab)
        batch_generator = BatchGenerator(text, self.batch_size, self.num_unroll, self.vocab_size, coder)

        for X in batch_generator:
            # forward prop
            caches_cache = []

            a0 = np.zeros((self.vocab_size, self.batch_size))
            a, y, c, caches = lstm_forward(X[:, :, :self.num_unroll], a0, self.parameters[0])
            caches_cache.append(caches)
            for layer in range(1, self.num_layers):
                a, y, c, caches = lstm_forward(y, a, self.parameters[layer])
                caches_cache.append(caches)

            loss = self.loss_func(y, X[:, :, 1:])

            gradients = lstm_backward(loss, caches_cache[len(caches_cache) - 1])
            update_weights(self.parameters[self.num_layers - 1][0], gradients, self.learning_rate)
            for layer in reversed(range(self.num_layers - 1)):
                gradients = lstm_backward(gradients['dx'], caches_cache[layer])
                update_weights(self.parameters[layer], gradients, self.learning_rate)

    def train_gpu(self, vocab, text):
        coder = VocabCoder(vocab)
        batch_generator = BatchGenerator(text, self.batch_size, self.num_unroll, self.vocab_size, coder)

        for X in batch_generator:
            # forward prop
            caches_cache = []

            a0 = np.zeros((self.vocab_size, self.batch_size))
            a, y, c, caches = lstm_forward_gpu(X[:, :, :self.num_unroll], a0, self.parameters[0])
            caches_cache.append(caches)
            for layer in range(1, self.num_layers):
                a, y, c, caches = lstm_forward_gpu(y, a, self.parameters[layer])
                caches_cache.append(caches)

            loss = self.loss_func(y, X[:, :, 1:])

            gradients = lstm_backward_gpu(loss, caches_cache[len(caches_cache) - 1])
            update_weights(self.parameters[self.num_layers - 1][0], gradients, self.learning_rate)
            for layer in reversed(range(self.num_layers - 1)):
                gradients = lstm_backward_gpu(gradients['dx'], caches_cache[layer])
                update_weights(self.parameters[layer], gradients, self.learning_rate)

    def run(self, vocab, seed):
        np.seed(seed)
        coder = VocabCoder(vocab)
        num_iter = 100

        X = np.zeros((self.vocab_size, self.batch_size, self.num_unroll))
        out = np.zeros(self.vocab_size, self.batch_size, num_iter)

        for i in range(self.batch_size):
            for j in range(self.num_unroll):
                X[randrange(0, self.vocab_size), i, j] = 1

        for i in range(num_iter):
            a0 = np.zeros((self.vocab_size, self.batch_size))
            a, y, c, caches = lstm_forward_gpu(X[:, :, :self.num_unroll], a0, self.parameters[0])
            for layer in range(1, self.num_layers):
                a, y, c, caches = lstm_forward_gpu(y, a, self.parameters[layer])
            out[:, :, i] = y

        for i in range(num_iter):
            for j in range(self.batch_size):
                print(coder.index_2_word(out[:, j, i]), end=" ")

    def loss_func(self, yhat, y):
        return y - yhat

    def allocate_parameters(self):
        parameters = []
        for i in range(self.num_layers):
            Wf = np.random.randn(self.layer_size[i], self.layer_size[i] + self.layer_size[i + 1])
            bf = np.random.randn(self.layer_size[i], 1)
            Wi = np.random.randn(self.layer_size[i], self.layer_size[i] + self.layer_size[i + 1])
            bi = np.random.randn(self.layer_size[i], 1)
            Wo = np.random.randn(self.layer_size[i], self.layer_size[i] + self.layer_size[i + 1])
            bo = np.random.randn(self.layer_size[i], 1)
            Wc = np.random.randn(self.layer_size[i], self.layer_size[i] + self.layer_size[i + 1])
            bc = np.random.randn(self.layer_size[i], 1)
            Wy = np.random.randn(self.layer_size[i], self.layer_size[i + 1])
            by = np.random.randn(self.layer_size[i], 1)
            parameters_cell = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo,
                               "bc": bc, "by": by}
            parameters.append(parameters_cell)
        return parameters

    """
    def serialize(self):
        pass

    def deserialize(self):
        pass
    """
