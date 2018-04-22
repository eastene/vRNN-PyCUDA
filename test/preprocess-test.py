import unittest
import numpy as np
from preprocess.nlp import top_k_word_frequencies, tokenize, normalize
from preprocess.Vocab import Vocab


class VocabTest(unittest.TestCase):

    def test_encoder(self):
        vocab = {"alpha", "beta", "charlie", "delta"}
        encoder = Vocab(vocab)
        words = ["alpha", "beta", "echo"]
        answer = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        i = 0
        for enc in encoder.encode(words):
            self.assertListEqual(enc.tolist(), answer[i])
            i += 1


class NLPTest(unittest.TestCase):

    def test_top_k_word_frequency(self):
        text = ["this", "this", "this", "is", "a", "is", "is", "test", "of", "of",
                "of", "the", "word", "counter", "word", "is"]
        k = 3
        answer = ["is", "of", "this"]
        self.assertCountEqual(top_k_word_frequencies(text, k), answer)

    def test_tokenize(self):
        text = "This is a test of the word tokenizer."
        tokens = ["This", "is", "a", "test", "of", "the", "word", "tokenizer", "."]

        self.assertCountEqual(tokenize(text), tokens)

    def test_normalize(self):
        tokens = ["This", "is", "a", "TEST", "of", "THe", "toKen", "normalizer", "."]
        answer = ["this", "is", "a", "test", "of", "the", "token", "normalizer", "."]

        self.assertListEqual(normalize(tokens), answer)


if __name__ == "__main__":
    unittest.main