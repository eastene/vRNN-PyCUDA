from src.model.Cell import Cell

import numpy as np

class Layer:

    def __init__(self, num_unroll, vocab_size, batch_size, layer_num):
        self.num_unroll = num_unroll
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.layer_num = layer_num

        self.state = np.zeros((batch_size, 1)) # state of last cell in layer (Tth cell)

        self.cells = [Cell(vocab_size, batch_size, (layer_num, i)) for i in range(num_unroll)]

    def forward_prop(self, train_input):
        for cell in self.cells:
            cell.forward_prop()