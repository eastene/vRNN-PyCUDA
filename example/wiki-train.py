import argparse
import json
import os


# parse single json object at a time
def parse_file(file):
    for line in file:
        yield json.loads(line)


def train():
    pass


def run():
    pass


def main():
    parser = argparse.ArgumentParser(description="Train or run RNN using example Wikipedia data.")
    parser.add_argument('--train', action='store_true', dest='train',
                        help='train on example datasets')
    parser.add_argument('--run', action='store_false', dest='train',
                        help='generate article text, requires trained model')
    parser.add_argument('--profile', dest='profile')
    parser.set_defaults(train=True)

    args = parser.parse_args()

    if args.train:
        train()
    else:
        run()


if __name__ == "__main__":
    main()