from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np


class Vocab:

    def __init__(self, vocab):
        self.vocab = vocab
        self.indexes = {}
        for i in range(len(vocab)):
            self.indexes[vocab[i]] = i
        self.vocab_size = len(vocab)

    def word_2_index(self, word):
        if word in self.indexes:
            return self.indexes[word]
        else:
            return -1

    def index_2_word(self, encoding):
        pass