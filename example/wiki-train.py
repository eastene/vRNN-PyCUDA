import argparse
import json
from model.RNN import RNN
import os
from example.vocab import top_k_word_frequencies


# parse single json object at a time
def parse_file(file):
    for line in file:
        yield json.loads(line)


def train(rnn):
    for f in os.listdir('train-example-data'):
        data = parse_file(f)
        for obj in data:
            rnn.train(obj['text'])


def run(rnn):
    pass


def profile(rnn):
    pass


def gen_vocab():
    for f in os.listdir('train-example-data'):
        data = parse_file(f)
        for obj in data:
            top_k_word_frequencies(obj['text'])

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
    parser.set_defaults(train=True, profile=False, vocab_size=10000)

    args = parser.parse_args()

    rnn = RNN(args.vocab_size, [100, 150])  # vocab size of 10000, 2 hidden layers of size 100

    if args.train:
        print("Beginning training on example dataset")
        train(rnn)
    else:
        run(rnn)


if __name__ == "__main__":
    main()