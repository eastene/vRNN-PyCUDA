import argparse
import string
import json
from src.model.LSTM import LSTM
import os
from src.preprocess.nlp import *

# parse single json object at a time
def parse_file(file):
    for line in file:
        yield json.loads(line)


def train(lstm, vocab, text, iterations, use_gpu, max_layers_on_gpu=1):
    if not use_gpu:
        lstm.train_gpu(vocab, text, iterations, max_layers_on_gpu)
    else:
        lstm.train(vocab, text, iterations)


def run(lstm, vocab, seed):
    lstm.run(vocab, seed)


def profile(rnn):
    pass


def gen_vocab(size):
    corpus = ""
    data = parse_file("wiki-train-data.txt")
    for obj in data:
        corpus += obj['text']
    return top_k_word_frequencies(corpus, size)


def read_vocab_file(vocab_file='example-vocab-10000.txt'):
    vocab = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            vocab += line
    return vocab



def main():
    parser = argparse.ArgumentParser(description="Train or run RNN using example Wikipedia data.")
    #parser.add_argument('--profile', action='store-true', dest='profile',
    #                    help='profile training/running RNN model save results to file')
    parser.add_argument('--vocab-size', dest='vocab_size', type=int,
                        help='specify vocabulary size, defaults to train on characters (default 27 tokens)')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='specify batch size for training (default 50 tokens)')
    parser.add_argument('--seq-len', dest='seq_len', type=int,
                        help='specify sequence length which is also number of LSTM cell unrollings (default 5)')
    parser.add_argument('--num-hidden-layers', dest='num_hidden_layers', type=int,
                        help='specify number of hidden layers to use (default 1)')
    parser.add_argument('--learn-rate', dest='learning_rate', type=float,
                        help='learning rate for updating the model')
    parser.add_argument('--iterations', dest='iterations', type=int,
                        help='max number of iterations to train the model (default = 10)')
    parser.add_argument('--seed', dest='seed', type=int,
                        help='seed the rnn input for sequence generation')
    parser.add_argument('--max-layer-gpu', dest='max_layer_gpu', type=int,
                        help='maximum number of LSTM layers allowed on GPU at a time (default = 2)')
    parser.add_argument('--force-cpu', action='store_true', dest='force_cpu',
                        help='WARNING: NOT RECOMMENDED - train on CPU only, can be used for sanity check of GPU results')
    parser.set_defaults(profile=False, vocab_size=27, batch_size=40, seq_len=5, num_hidden_layers=2,
                        learning_rate=0.5, iterations=10, seed=42, max_layer_gpu=1, force_cpu=False)

    args = parser.parse_args()

    lstm = LSTM(args.seq_len, args.vocab_size, args.batch_size, args.num_hidden_layers, args.learning_rate)  # input/output layer required

    # Step 1: Vocab generation
    if args.vocab_size == 27:
        print("Using 27 tokens (ascii lowercase and whitespace).")
        vocab = string.ascii_lowercase + " "
    else:
        print("Generating vocabulary of " + str(args.vocab_size) + " unique tokens.")
        vocab = gen_vocab(args.vocab_size)

    # Step 2: NLP processing of corpus
    with open('wiki-train-data.txt', 'r') as f:
        training_set = f.read()

    if args.vocab_size == 27:
        tokens = tokenize_char(training_set)
    else:
        tokens = tokenize(training_set)

    normal = normalize(tokens)

    # Step 3: Encoding and RNN training
    print("Beginning training on example dataset")
    train(lstm, vocab, normal, args.iterations, args.force_cpu, args.max_layer_gpu)

    run(lstm, vocab, normalize(tokenize_char("Apple's first logo, designed by Ron Wayne, depicts Sir Isaac Newton sitting under an apple tree")))


if __name__ == "__main__":
    main()