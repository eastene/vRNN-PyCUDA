import argparse
import string
import json
from src.model.LSTM import LSTM
import os
from src.preprocess.nlp import *
from time import time

# parse single json object at a time for data in JSON
def parse_file(file):
    for line in file:
        yield json.loads(line)


def train(lstm, vocab, text, iterations, use_gpu, max_layers_on_gpu):
    if not use_gpu:
        if max_layers_on_gpu == lstm.num_layers or max_layers_on_gpu == 1:
            lstm.train_gpu(vocab, text, iterations, max_layers_on_gpu)
        else:
            lstm.train_gpu_async(vocab, text, iterations, max_layers_on_gpu)
    else:
        lstm.train(vocab, text, iterations)


def run(lstm, vocab, seed):
    lstm.run(vocab, seed)


def profile():
    """
    Profile runtimes of 3 methods of GPU training with 5 different sizes.
        1 - All layers on GPU at all times
        2 - Prefetch layers 1 ahead with 2 layers on GPU at a time
        3 - No prefetching, 2 layers on GPU at a time
    :return: None
    """
    print("Beginning Profiling...")
    # default profiling parameters
    seq_len = 5
    batch_size = 5
    num_hidden_layers = 3
    learning_rate = 0.5

    times = []

    with open('wiki-train-data.txt', 'r') as f:
        training_set = f.read()

    tokens = tokenize(training_set)
    normal = normalize(tokens)

    print("Generating vocabs for profiling... ")
    vocabs = [gen_vocab(normal, 10)]
    print("Generating vocabs for profiling... (1/5)")
    vocabs.append(gen_vocab(normal, 100))
    print("Generating vocabs for profiling... (2/5)")
    vocabs.append(gen_vocab(normal, 500))
    print("Generating vocabs for profiling... (3/5)")
    vocabs.append(gen_vocab(normal, 1000))
    print("Generating vocabs for profiling... (4/5)")
    vocabs.append(gen_vocab(normal, 1500))
    print("Generating vocabs for profiling... (5/5)")

    for vocab in vocabs:

        # Round 1 - All layers on GPU vs prefecthing
        lstm = LSTM(seq_len, len(vocab), batch_size, num_hidden_layers, learning_rate)
        s1 = time()
        lstm.train_gpu(vocab, normal, 3, num_hidden_layers)
        e1 = time()

        lstm = LSTM(seq_len, len(vocab), batch_size, num_hidden_layers, learning_rate)
        s2 = time()
        lstm.train_gpu_async(vocab, normal, 3, 2)
        e2 = time()

        # Round 2 - sequential prefetching vs async prefetching
        lstm = LSTM(seq_len, len(vocab), batch_size, num_hidden_layers, learning_rate)
        s3 = time()
        lstm.train_gpu(vocab, normal, 3, 2)
        e3 = time()

        times.append((e1, s1, e2, s2, e3, s3))

    for i in range(len(vocabs)):
        print("For vocab size of: {0}".format(len(vocabs[i])))
        e1, s1, e2, s2, e3, s3 = times[i]
        print("No Prefetching, all layers: {0}".format(e1 - s1))
        print("Prefetching enabled, 2 layers: {0}".format(e2 - s2))
        print("No Prefetching, 2 layers: {0}".format(e3 - s3))


def gen_vocab(tokens, size):
    if size < 1500:
        return top_k_word_frequencies(tokens, size)
    else:
        return sample_k_words(tokens, size)


def read_vocab_file(vocab_file='example-vocab-10000.txt'):
    vocab = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            vocab += line
    return vocab


def main():
    parser = argparse.ArgumentParser(description="Train or run RNN using example Wikipedia data.")
    parser.add_argument('--profile', action='store_true', dest='profile',
                        help='profile training/running RNN model save results to file')
    parser.add_argument('--vocab-size', dest='vocab_size', type=int,
                        help='specify vocabulary size, defaults to train on characters (default 27 tokens)')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='specify batch size for training (default 50 tokens)')
    parser.add_argument('--seq-len', dest='seq_len', type=int,
                        help='specify sequence length which is also number of LSTM cell unrollings (default 5)')
    parser.add_argument('--num-layers', dest='num_hidden_layers', type=int,
                        help='specify number of layers to use (default 4)')
    parser.add_argument('--learn-rate', dest='learning_rate', type=float,
                        help='learning rate for updating the model')
    parser.add_argument('--iterations', dest='iterations', type=int,
                        help='max number of iterations to train the model (default = 10)')
    parser.add_argument('--seed', dest='seed', type=int,
                        help='seed the rnn input for sequence generation')
    parser.add_argument('--max-layer-gpu', dest='max_layer_gpu', type=int,
                        help='maximum number of LSTM layers allowed on GPU at a time, to enable layer prefetching, '
                             'must be 2 or more to allow a layer to be prefetched while executing the current layer,'
                             'if set to 1, or equal to the number of layers, no prefetching will be used(default = 2)')
    parser.add_argument('--force-cpu', action='store_true', dest='force_cpu',
                        help='WARNING: NOT RECOMMENDED - train on CPU only, can be used for sanity check of GPU results')
    parser.set_defaults(profile=False, vocab_size=27, batch_size=50, seq_len=5, num_hidden_layers=4,
                        learning_rate=0.5, iterations=10, seed=42, max_layer_gpu=2, force_cpu=False)

    args = parser.parse_args()

    if args.profile:
        profile()
        return


    # Step 1: NLP processing of corpus
    with open('wiki-train-data.txt', 'r') as f:
        training_set = f.read().replace('\n', ' ')

    if args.vocab_size == 27:
        tokens = tokenize_char(training_set)
    else:
        tokens = tokenize(training_set)

    normal = normalize(tokens)

    # Step 2: Vocab generation
    if args.vocab_size == 27:
        print("Using 27 tokens (ascii lowercase and whitespace).")
        vocab = string.ascii_lowercase + " "
    else:
        print("Generating vocabulary of " + str(args.vocab_size) + " unique tokens.")
        vocab = gen_vocab(normal, args.vocab_size)

    # Step 3: Encoding and RNN training
    print("Beginning training on example dataset")
    lstm = LSTM(args.seq_len, args.vocab_size, args.batch_size, args.num_hidden_layers,
                args.learning_rate)  # input/output layer required
    s = time()
    train(lstm, vocab, normal, args.iterations, args.force_cpu, args.max_layer_gpu)
    e = time()

    print("Training complete in {0} s on vocab of {1} tokens".format((e - s), len(vocab)))

    if args.vocab_size == 27:
        run(lstm, vocab, normalize(tokenize_char(
            "Apple's first logo, designed by Ron Wayne, depicts Sir Isaac Newton sitting under an apple tree"))
        )
    else:
        run(lstm, vocab, normalize(tokenize(
            "Apple's first logo, designed by Ron Wayne, depicts Sir Isaac Newton sitting under an apple tree"))
        )


if __name__ == "__main__":
    main()