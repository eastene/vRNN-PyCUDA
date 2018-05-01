#from pycuda.tools import make_default_context
import pycuda.autoinit
from skcuda.misc import init as skcuda_init
from sklearn.metrics import log_loss
from textwrap import wrap
from time import time

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
        self.gpu_parameters = []

    def train(self, vocab, text, iterations):
        coder = VocabCoder(vocab)
        batch_generator = BatchGenerator(text, self.batch_size, self.num_unroll, self.vocab_size, coder)

        for i in range(iterations):

            X = batch_generator.gen_batch()

            # forward prop
            caches_cache = []

            a0 = np.zeros((self.vocab_size, self.batch_size))
            a, y, c, caches = lstm_forward(X[:, :, :self.num_unroll], a0, self.parameters[0])
            caches_cache.append(caches)
            for layer in range(1, self.num_layers):
                a, y, c, caches = lstm_forward(y, a0, self.parameters[layer])
                caches_cache.append(caches)

            loss = self.loss_func(X[:, :, 1:], y)

            gradients = lstm_backward(loss, caches_cache[len(caches_cache) - 1])
            update_weights(self.parameters[self.num_layers - 1], gradients, self.learning_rate)
            for layer in reversed(range(self.num_layers - 1)):
                gradients = lstm_backward(gradients['dx'], caches_cache[layer])
                update_weights(self.parameters[layer], gradients, self.learning_rate)

    def train_gpu(self, vocab, text, iterations, allowable_layers_on_gpu=1):
        skcuda_init()
        coder = VocabCoder(vocab)
        batch_generator = BatchGenerator(text, self.batch_size, self.num_unroll, self.vocab_size, coder)
        layer_queue = []

        print("")

        for i in range(iterations):

            print("Iteration: {0}  --  starting at: {1}".format(i, time()))

            X = batch_generator.gen_batch()

            caches_cache = []

            # forward prop
            a0 = np.zeros((self.vocab_size, self.batch_size))
            a, y, c, caches = self.request_layer_transfer_forward(0, X[:, :, :self.num_unroll], a0, layer_queue, allowable_layers_on_gpu)
            caches_cache.append(caches)
            for layer in range(1, self.num_layers):
                a, y, c, caches = self.request_layer_transfer_forward(layer, y, a0, layer_queue, allowable_layers_on_gpu)
                caches_cache.append(caches)

            loss = self.loss_func(X[:, :, 1:], y.get())

            gradients = lstm_backward_gpu(loss, caches_cache[len(caches_cache) - 1])
            self.request_layer_transfer_back(self.num_layers - 1, gradients, layer_queue, allowable_layers_on_gpu)
            for layer in reversed(range(self.num_layers - 1)):
                gradients = lstm_backward_gpu(gradients['dx'], caches_cache[layer])
                self.request_layer_transfer_back(layer, gradients, layer_queue, allowable_layers_on_gpu)

        self.evict_gpu(layer_queue)

    def request_layer_transfer_forward(self, layer_num, X, a, layer_queue, max_layers_on_gpu):
        if len(layer_queue) >= max_layers_on_gpu and layer_num not in layer_queue:
            evict = layer_queue.pop(0)
            print("Evicting layer {0} from GPU".format(evict))
            self.parameters[evict] = layer_from_gpu(self.parameters[evict])
        if layer_num not in layer_queue:
            self.parameters[layer_num] = layer_to_gpu(self.parameters[layer_num])
            layer_queue.append(layer_num)
            print("Placing layer {0} onto GPU for Forward Prop".format(layer_num))
        print("Layers on GPU: {0}".format(layer_queue))
        return lstm_forward_gpu(X, a, self.parameters[layer_num])

    def request_layer_transfer_back(self, layer_num, gradients, layer_queue, max_layers_on_gpu):
        if len(layer_queue) >= max_layers_on_gpu and layer_num not in layer_queue:
            evict = layer_queue.pop(0)
            print("Evicting layer {0} from GPU".format(evict))
            self.parameters[evict] = layer_from_gpu(self.parameters[evict])
        if layer_num not in layer_queue:
            self.parameters[layer_num] = layer_to_gpu(self.parameters[layer_num])
            layer_queue.append(layer_num)
            print("Placing layer {0} onto GPU for Back Prop".format(layer_num))
        print("Layers on GPU: {0}".format(layer_queue))
        update_weights(self.parameters[layer_num], gradients, self.learning_rate)

    def evict_gpu(self, layer_queue):
        print("Training Done: Evicting GPU")
        while len(layer_queue) > 0:
            evict = layer_queue.pop(0)
            try:
                self.parameters[evict] = layer_from_gpu(self.parameters[evict])
            except AttributeError:
                print("Layer already on CPU")

    def run(self, vocab, seed):
        print("Generating Output:\n\n")
        coder = VocabCoder(vocab)
        num_iter = 100

        X = np.zeros((self.vocab_size, self.batch_size, self.num_unroll))
        a0 = np.zeros((self.vocab_size, self.batch_size))
        out = []

        k = 0
        for i in range(self.batch_size):
            for j in range(self.num_unroll):
                X[coder.word_2_index(seed[k]), i, j] = 1
                k = (k + 1) % len(seed)

        for i in range(num_iter):
            a, y, c, caches = lstm_forward(X[:, :, :self.num_unroll], a0, self.parameters[0])
            for layer in range(1, self.num_layers):
                a, y, c, caches = lstm_forward(y, a0, self.parameters[layer])

            for i in range(self.num_unroll):
                for j in range(self.batch_size):
                    out.append(coder.index_2_word(np.argmax(np.transpose(y[:, j, i]))))
            X = y

        print("\n".join(wrap("".join(out), 80)))

    def loss_func(self, y, yhat):
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
