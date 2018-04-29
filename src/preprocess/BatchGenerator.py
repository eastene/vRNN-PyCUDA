import numpy as np

from src.preprocess.VocabCoder import VocabCoder


class BatchGenerator:

    def __init__(self, text, batch_size, num_unroll, vocab_size, vocab_coder: VocabCoder):
        self.text = text
        self.batch_size = batch_size
        self.num_unroll = num_unroll
        self.vocab_size = vocab_size
        self.vocab_coder = vocab_coder

        self.num_batches = len(text) // batch_size
        self.bookmark = 0

    def __iter__(self):
        return self

    def __next__(self):
        yield self.gen_batch()

    def gen_batch(self):
        batch = np.zeros((self.num_unroll, self.vocab_size, self.batch_size))
        for i in range(self.batch_size):
            idx = self.vocab_coder.word_2_index(self.text[(self.bookmark + i) % len(self.text)])
            if idx != -1:
                batch[i][idx] = 1
        self.bookmark = (self.bookmark + self.batch_size) % len(self.text)

        return batch[:self.batch_size - 2], batch[1:]