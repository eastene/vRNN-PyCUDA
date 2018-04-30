import numpy as np

from src.model.Cell import Cell


class Layer:

    def __init__(self, num_unroll, vocab_size, batch_size, layer_num):
        self.num_unroll = num_unroll
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.layer_num = layer_num

        self.output = np.zeros((batch_size, 1))
        self.state = np.zeros((batch_size, 1))  # state of last cell in layer (Tth cell)

        self.cell = Cell(vocab_size, batch_size, (layer_num, 0))

    def forward_prop(self, train_input):
        h = np.zeros(self.vocab_size, self.batch_size)
        c = np.zeros(self.vocab_size, self.batch_size)
        out = [h]
        for t in range(self.num_unroll):
            h, c = self.cell.forward_prop(h, c, train_input[t])
            out.append(h)

        self.state = c
        self.output = out

        return out