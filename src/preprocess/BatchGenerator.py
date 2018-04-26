import numpy as np

from src.preprocess.Vocab import Vocab


class BatchGenerator:

    def __init__(self, text, batch_size, vocab_size, vocab_coder: Vocab):
        self.text = text
        self.batch_size = batch_size + 1  # output is shifted by 1
        self.vocab_size = vocab_size
        self.vocab_coder = vocab_coder

        self.num_batches = len(text) // batch_size
        self.bookmark = 0

    def gen_batch(self):
        batch = np.zeros((self.vocab_size, self.batch_size))
        idx = 0
        for i in range(self.batch_size):
            idx = self.vocab_coder.word_2_index(self.text[(self.bookmark + i) % len(self.text)])
            if idx != -1:
                batch[i][idx] = 1
        self.bookmark = (self.bookmark + self.batch_size) % len(self.text)

        return batch[:self.batch_size - 2], batch[1:]

    def __next__(self):
        yield self.gen_batch()

