import argparse
import json
from src.model.RNN import RNN
import os
from src.preprocess.nlp import top_k_word_frequencies, tokenize, normalize
from src.preprocess.Vocab import Vocab

# parse single json object at a time
def parse_file(file):
    for line in file:
        yield json.loads(line)


def train(rnn, batch_generator):
    rnn.train(batch_generator)


def run(rnn, seed):
    pass


def profile(rnn):
    pass


def gen_vocab(size, vocab_file='example-vocab.txt'):
    corpus = ""
    for f in os.listdir('train-example-data'):
        data = parse_file(f)
        for obj in data:
            corpus += obj['text']
    top_k = top_k_word_frequencies(corpus, size)

    with open(vocab_file, 'w') as f:
        for word in top_k:
            f.write(word)
        f.write("<UNK>")
        f.write("<START>")
        f.write("<STOP>")


def read_vocab_file(vocab_file='example-vocab-10000.txt'):
    vocab = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            vocab += line
    return vocab



def main():
    parser = argparse.ArgumentParser(description="Train or run RNN using example Wikipedia data.")
    parser.add_argument('--train', action='store_true', dest='train',
                        help='train on example datasets')
    parser.add_argument('--run', action='store_false', dest='train',
                        help='generate article text, requires trained model')
    parser.add_argument('--profile', action='store_true', dest='profile',
                        help='profile training/running RNN model save results to file')
    parser.add_argument('--vocabsize', dest='vocab_size',
                        help='specify vocabulary size (default 10000 tokens), only applies when training')
    parser.add_argument('--seed', dest='seed',
                        help='seed the rnn input for sequence generation')
    parser.add_argument('--force-cpu', action='store-true', dest='force_cpu',
                        help='WARNING: NOT RECOMMENDED - train on CPU only, can be used for sanity check of GPU results')
    parser.set_defaults(train=True, profile=False, vocab_size=10000, seed=42, force_cpu=False)

    args = parser.parse_args()

    rnn = RNN(args.vocab_size, [100, 100])  # vocab size of 10000, 2 hidden layers of size 100
    if args.train:
        # Step 1: Vocab generation
        print("Generating vocabulary of " + str(args.vocab_size) + " unique tokens.")
        vocab = {}
        if args.vocab_size == 10000:
            vocab = read_vocab_file()
        else:
            vocab = read_vocab_file('example-vocab-' + str(args.vocab_size) + '.txt')

        # Step 2: NLP processing of corpus
        training_set = ""
        for f in os.listdir('train-example-data'):
            data = parse_file(f)
            for obj in data:
                training_set += obj['text']
        tokens = tokenize(training_set)
        normal = normalize(tokens)

        # Step 3: Encoding and RNN training
        encoder = Vocab(vocab)
        print("Beginning training on example dataset")
        train(rnn, encoder.encode(normal))

    else:
        run(rnn, args.seed)


if __name__ == "__main__":
    main()