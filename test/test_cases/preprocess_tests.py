import unittest
import numpy as np
from src.preprocess.nlp import top_k_word_frequencies, tokenize, normalize
from src.preprocess.VocabCoder import VocabCoder
from src.preprocess.BatchGenerator import BatchGenerator

class BatchGeneratorTestCase(unittest.TestCase):

    def test_iter(self):
        text = "This is a test of the encoder"
        tokens = tokenize(text)
        normal = normalize(tokens)

        vocab = ["this", "is", "test", "of", "the"]
        coder = VocabCoder(vocab)
        batcher = BatchGenerator(normal, 2, 2, 5, coder)

        for batch in batcher:
            print(batch)

    def test_batch(self):
        text = "This is a test of the encoder"
        tokens = tokenize(text)
        normal = normalize(tokens)

        vocab = ["this", "is", "test", "of", "the"]
        coder = VocabCoder(vocab)
        batcher = BatchGenerator(normal, 2, 2, 5, coder)
        batch = batcher.gen_batch()

        answer = np.array([[1,0,0,0,0], [0,1,0,0,0]])

        print(np.allclose(np.transpose(batch[:,:,0]), answer))

class VocabTestCase(unittest.TestCase):

    def test_encoder(self):
        vocab = ["alpha", "beta", "charlie", "delta"]
        encoder = VocabCoder(vocab)
        words = ["alpha", "beta", "echo"]
        answer = [0, 1, -1]
        for i in range(len(words)):
            enc = encoder.word_2_index(words[i])
            self.assertEqual(enc, answer[i])


class NLPTestCase(unittest.TestCase):

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