from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np


class Vocab:

    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.le = LabelEncoder()
        encoding = self.le.fit_transform(list(vocab))

        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(encoding.reshape(-1, 1))

    def encode(self, words):
        for word in words:
            if word in self.vocab:
                enc = self.le.transform([word]).reshape(-1, 1)
                yield self.ohe.transform(enc).toarray().astype(np.int64)[0]

            # unknown word
            else:
                yield np.array([0 for i in range(self.vocab_size)])