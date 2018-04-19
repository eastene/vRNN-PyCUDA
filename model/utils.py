import math
from sklearn.utils import extmath


def sigmoid(X):
    return 1 / (1 + math.exp(-X))


def tanh(X):
    return math.tanh(X)


def softmax(X):
    return extmath.softmax(X)