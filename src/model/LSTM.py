#from pycuda.tools import make_default_context
import pycuda.autoinit
import pycuda.driver
import skcuda.linalg
from textwrap import wrap
from time import time

from src.model.lstm_layer import *
from src.preprocess.BatchGenerator import BatchGenerator
from src.preprocess.VocabCoder import VocabCoder

LOGGING = True
LOG_INTERVAL = 25

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
        self.gpu_streams = {}

        # fully connected output layer
        self.Wy = np.random.uniform(-1, 1, (self.layer_size[-1], self.vocab_size))
        self.by = np.random.uniform(-0.01, 0.01, (1, self.vocab_size))

    def train(self, vocab, text, iterations):
        coder = VocabCoder(vocab)
        batch_generator = BatchGenerator(text, self.batch_size, self.num_unroll, self.vocab_size, coder)

        for i in range(iterations):

            X = batch_generator.gen_batch()

            # forward prop
            caches_cache = []
            a0 = np.zeros((self.batch_size, self.layer_size[0]))
            a, c, caches = lstm_forward(X, a0, self.parameters[0])
            caches_cache.append(caches)
            for layer in range(1, self.num_layers):
                a, c, caches = lstm_forward(a, a0, self.parameters[layer])
                caches_cache.append(caches)

            loss = []
            probabilities = []
            # compute loss from fully connected layer
            for roll in range(self.num_unroll):
                output_layer = np.matmul(a[:, :, roll], self.Wy) + self.by
                softmax_layer = softmax(output_layer)
                probabilities.append(softmax_layer)
                loss.append(self.loss_func(X[:, :, 1+roll], softmax_layer))

            if LOGGING and (i % LOG_INTERVAL == 0):
                print("Loss={}, step={}".format(loss, i))
                print("Probabilities=")
                for roll in range(self.num_unroll):
                    print(probabilities[i])

            dWy = np.zeros_like(self.Wy)
            da = np.zeros_like(a)
            dby = np.zeros_like(self.by)

            for i in range(self.num_unroll):
                print(a.shape)
                dWy += loss[i] * a[:, :, i]
                dby += loss[i]

            self.Wy = self.Wy + dWy
            self.by = self.by + dby

            for i in range(self.num_unroll):
                da[:, :, i] = loss[i] * self.Wy

            gradients = lstm_backward(da, caches_cache[len(caches_cache) - 1])
            update_weights(self.parameters[self.num_layers - 1], gradients, self.learning_rate)
            for layer in reversed(range(self.num_layers - 1)):
                gradients = lstm_backward(gradients['dx'], caches_cache[layer])
                update_weights(self.parameters[layer], gradients, self.learning_rate)

    def train_gpu(self, vocab, text, iterations, allowable_layers_on_gpu=1):
        print("Training LSTM on GPU with Prefetching disabled")
        skcuda.linalg.init()
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

            dx = self.request_layer_transfer_back(self.num_layers - 1, loss, caches_cache, layer_queue, allowable_layers_on_gpu)
            for layer in reversed(range(self.num_layers - 1)):
                dx = self.request_layer_transfer_back(layer, dx, caches_cache, layer_queue,
                                                 allowable_layers_on_gpu)

        self.evict_gpu(layer_queue)

    def train_gpu_async(self, vocab, text, iterations, allowable_layers_on_gpu=2):
        """
        This function trains the RNN model using the GPU and layer prefetching is enabled. Layer prefetching will
        use an asynchronous memory transfer to transfer the model parameters while the GPU operates on the current
        layer.
        :param vocab:
        :param text:
        :param iterations:
        :param allowable_layers_on_gpu:
        :return:
        """
        print("Training LSTM on GPU with Prefetching enabled")
        skcuda.linalg.init()
        coder = VocabCoder(vocab)
        batch_generator = BatchGenerator(text, self.batch_size, self.num_unroll, self.vocab_size, coder)
        layer_queue = []

        print("")

        # run once to initalize 1st layer
        self.initialize_layer_transfer_forward(0, layer_queue, allowable_layers_on_gpu)

        for i in range(iterations):

            print("Iteration: {0}  --  starting at: {1}".format(i, time()))

            X = batch_generator.gen_batch()

            caches_cache = []

            # forward prop
            a0 = np.zeros((self.vocab_size, self.batch_size))

            a, y, c, caches = self.prefetch_layer_forward(0, X[:, :, :self.num_unroll], a0, layer_queue, allowable_layers_on_gpu)
            caches_cache.append(caches)
            for layer in range(1, self.num_layers):
                a, y, c, caches = self.prefetch_layer_forward(layer, y, a0, layer_queue, allowable_layers_on_gpu)
                caches_cache.append(caches)

            loss = self.loss_func(X[:, :, 1:], y.get())

            # backprop requires reversing the layer queue since layers on the GPU already can be used
            layer_queue = list(reversed(layer_queue))

            gradients = lstm_backward_gpu(loss, caches_cache[len(caches_cache) - 1])
            self.prefetch_layer_back(self.num_layers - 1, gradients, layer_queue, allowable_layers_on_gpu)
            for layer in reversed(range(self.num_layers - 1)):
                gradients = lstm_backward_gpu(gradients['dx'], caches_cache[layer])
                self.prefetch_layer_back(layer, gradients, layer_queue, allowable_layers_on_gpu)

            # return queue to normal for forward prop
            layer_queue = list(reversed(layer_queue))

        self.evict_gpu(layer_queue)

    def prefetch_layer_forward(self, layer_num, X, a, layer_queue, max_layers_on_gpu):
        """
        This function handles the allocation/deallocation of each RNN layer on the GPU's main memory
        it uses a queue to determine when to evict a layer and when to add a layer. In the original
        paper, which is specifically meant for CNNs, the next layer requires a function to prefetch
        the layer's data to the GPU this is because the layers are not always laid out sequentially in a CNN.
        Since our LSTM model is entirely sequential in the layers, we can avoid needing to calculate the layer to
        prefetch, since it will always be the next in sequence.

        This function works for forwards propagation in which layers are accessed in the order 0, 1, ..., N
        :param layer_num: which layer is being placed on the GPU
        :param X: data used in forward prop or output of
        :param a: initial bias (all 0s)
        :param layer_queue: order of layers on GPU
        :param max_layers_on_gpu: Most layers allowable on GPU at a time
        :return: function for forwards propagation
        """
        if layer_num < self.num_layers - 1:
            strm1 = pycuda.driver.Stream()
            strm2 = pycuda.driver.Stream()
            next_layer = layer_num + 1
            if len(layer_queue) >= max_layers_on_gpu and next_layer not in layer_queue:
                evict = layer_queue.pop(0)
                print("Evicting layer {0} from GPU".format(evict))
                self.parameters[evict] = layer_from_gpu_async(self.parameters[evict], strm1)
            if next_layer not in layer_queue:
                print("Prefetching layer {0} onto GPU for Forward Prop".format(next_layer))
                self.parameters[next_layer] = layer_to_gpu_async(self.parameters[next_layer], strm2)
                layer_queue.append(next_layer)
                self.gpu_streams[next_layer] = strm2
        print("Layers on GPU: {0}".format(layer_queue))
        if layer_num in self.gpu_streams and not self.gpu_streams[layer_num].is_done():
            print("Waiting for layer {0} to finish fetching".format(layer_num))
            self.gpu_streams[layer_num].synchronize()
            del self.gpu_streams[layer_num]
        return lstm_forward_gpu(X, a, self.parameters[layer_num])

    def prefetch_layer_back(self, layer_num, gradients, layer_queue, max_layers_on_gpu):
        """
        This function handles the allocation/deallocation of each RNN layer on the GPU's main memory
        it uses a queue to determine when to evict a layer and when to add a layer. In the original
        paper, which is specifically meant for CNNs, the next layer requires a function to prefetch
        the layer's data to the GPU this is because the layers are not always laid out sequentially in a CNN.
        Since our LSTM model is entirely sequential in the layers, we can avoid needing to calculate the layer to
        prefetch, since it will always be the next in sequence.

        This function works for backwards propagation in which layers are accessed in the order N, N - 1, ..., 0
        :param layer_num: which layer to compute (prefetch layer_num - 1th layer)
        :param gradients: ouput of backward propagation for the current layer
        :param layer_queue: order of layers on GPU
        :param max_layers_on_gpu: Most layers allowable on GPU at a time
        :return: None
        """
        if layer_num > 0:
            strm1 = pycuda.driver.Stream()
            strm2 = pycuda.driver.Stream()
            next_layer = layer_num - 1
            if len(layer_queue) >= max_layers_on_gpu and next_layer not in layer_queue:
                evict = layer_queue.pop(0)
                print("Evicting layer {0} from GPU".format(evict))
                self.parameters[evict] = layer_from_gpu_async(self.parameters[evict], strm1)
            if next_layer not in layer_queue:
                print("Prefetching layer {0} onto GPU for Back Prop".format(next_layer))
                self.parameters[next_layer] = layer_to_gpu_async(self.parameters[next_layer], strm2)
                layer_queue.append(next_layer)
                self.gpu_streams[next_layer] = strm2
        print("Layers on GPU: {0}".format(layer_queue))
        if layer_num in self.gpu_streams and not self.gpu_streams[layer_num].is_done():
            print("Waiting for layer {0} to finish fetching".format(layer_num))
            self.gpu_streams[layer_num].synchronize()
            del self.gpu_streams[layer_num]
        update_weights(self.parameters[layer_num], gradients, self.learning_rate)

    def initialize_layer_transfer_forward(self, layer_num, layer_queue, max_layers_on_gpu):
        if len(layer_queue) >= max_layers_on_gpu and layer_num not in layer_queue:
            evict = layer_queue.pop(0)
            print("Evicting layer {0} from GPU".format(evict))
            self.parameters[evict] = layer_from_gpu(self.parameters[evict])
        if layer_num not in layer_queue:
            print("Placing layer {0} onto GPU for Forward Prop".format(layer_num))
            self.parameters[layer_num] = layer_to_gpu(self.parameters[layer_num])
            layer_queue.append(layer_num)
        print("Layers on GPU: {0}".format(layer_queue))

    def request_layer_transfer_forward(self, layer_num, X, a, layer_queue, max_layers_on_gpu):
        if len(layer_queue) >= max_layers_on_gpu and layer_num not in layer_queue:
            evict = layer_queue.pop(0)
            print("Evicting layer {0} from GPU".format(evict))
            self.parameters[evict] = layer_from_gpu(self.parameters[evict])

        if layer_num not in layer_queue:
            print("Placing layer {0} onto GPU for Forward Prop".format(layer_num))
            self.parameters[layer_num] = layer_to_gpu(self.parameters[layer_num])
            layer_queue.append(layer_num)
        print("Layers on GPU: {0}".format(layer_queue))
        return lstm_forward_gpu(X, a, self.parameters[layer_num])

    def request_layer_transfer_back(self, layer_num, loss, cache, layer_queue, max_layers_on_gpu):
        if len(layer_queue) >= max_layers_on_gpu and layer_num not in layer_queue:
            evict = layer_queue.pop(0)
            print("Evicting layer {0} from GPU".format(evict))
            self.parameters[evict] = layer_from_gpu(self.parameters[evict])

        if layer_num not in layer_queue:
            print("Placing layer {0} onto GPU for Back Prop".format(layer_num))
            self.parameters[layer_num] = layer_to_gpu(self.parameters[layer_num])
            layer_queue.append(layer_num)
        print("Layers on GPU: {0}".format(layer_queue))
        gradients = lstm_backward_gpu(loss, cache[layer_num])
        update_weights(self.parameters[layer_num], gradients, self.learning_rate)
        return gradients['dx']

    def evict_gpu(self, layer_queue):
        print("Training Done: Evicting GPU")
        while len(layer_queue) > 0:
            evict = layer_queue.pop(0)
            try:
                self.parameters[evict] = layer_from_gpu(self.parameters[evict])
            except AttributeError:
                print("Layer already on CPU")

    def run(self, vocab, seq_seed):
        print("Generating Output:\n\n")
        coder = VocabCoder(vocab)
        num_iter = 100

        X = np.zeros((self.vocab_size, self.batch_size, self.num_unroll))
        a0 = np.zeros((self.vocab_size, self.batch_size))
        out = []

        k = 0
        for i in range(self.batch_size):
            for j in range(self.num_unroll):
                X[coder.word_2_index(seq_seed[k]), i, j] = 1
                k = (k + 1) % len(seq_seed)

        for i in range(num_iter):
            a, y, c, caches = lstm_forward(X[:, :, :self.num_unroll], a0, self.parameters[0])
            for layer in range(1, self.num_layers):
                a, y, c, caches = lstm_forward(y, a0, self.parameters[layer])

            for j in range(self.num_unroll):
                y_seq = self.Wy * y[:, :, j] + self.by
                for k in range(self.batch_size):
                    #if i == 1 and j == 1 and k == 1:
                    out.append(coder.index_2_word(np.argmax(np.transpose((y_seq[:, k])))))
            X = y

        print("\n".join(wrap("".join(out), 80)))

    def loss_func(self, y, output):
        y_hat = output
        y_true = []
        for j in range(self.batch_size):
            y_true.append(np.argmax(np.transpose((y[:, j]))))
        print(y_hat[range(self.batch_size), np.hstack(y_true)])
        log_loss = -np.log(y_hat[range(self.batch_size), np.hstack(y_true)])
        return np.sum(log_loss) / self.batch_size

    def allocate_parameters(self):
        parameters = []
        for i in range(self.num_layers):
            # generally of form (x :: a)A + b
            Wf = np.random.uniform(-1, 1, ((self.layer_size[i] + self.layer_size[i + 1]), self.layer_size[i + 1]))
            bf = np.random.uniform(-0.1, 0.1, (1, self.layer_size[i + 1]))
            Wi = np.random.uniform(-1, 1, ((self.layer_size[i] + self.layer_size[i + 1]), self.layer_size[i + 1]))
            bi = np.random.uniform(-0.1, 0.1, (1, self.layer_size[i + 1]))
            Wo = np.random.uniform(-1, 1, ((self.layer_size[i] + self.layer_size[i + 1]), self.layer_size[i + 1]))
            bo = np.random.uniform(-0.1, 0.1, (1, self.layer_size[i + 1]))
            Wc = np.random.uniform(-1, 1, ((self.layer_size[i] + self.layer_size[i + 1]), self.layer_size[i + 1]))
            bc = np.random.uniform(-0.1, 0.1, (1, self.layer_size[i + 1]))

            parameters_cell = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "bf": bf, "bi": bi, "bo": bo,
                               "bc": bc}
            parameters.append(parameters_cell)
        return parameters

    """
    def serialize(self):
        pass

    def deserialize(self):
        pass
    """
