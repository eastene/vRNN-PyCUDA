import argparse
import json
from model.RNN import RNN
import os


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


def main():
    parser = argparse.ArgumentParser(description="Train or run RNN using example Wikipedia data.")
    parser.add_argument('--train', action='store_true', dest='train',
                        help='train on example datasets')
    parser.add_argument('--run', action='store_false', dest='train',
                        help='generate article text, requires trained model')
    parser.add_argument('--profile', dest='profile',
                        help='profile training/running RNN model save results to file')
    parser.set_defaults(train=True)

    args = parser.parse_args()

    rnn = RNN(10000, [100, 150])  # vocab size of 10000, 2 hidden layers of size 100

    if args.train:
        print("Beginning training on example dataset")
        train(rnn)
    else:
        run(rnn)


if __name__ == "__main__":
    main()