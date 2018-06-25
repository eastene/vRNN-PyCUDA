import math

import numpy as np
from sklearn.utils import extmath


def sigmoid(X):
    func = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
    return func(X)


def tanh(X):
    func = np.vectorize(lambda x: math.tanh(x))
    return func(X)


def softmax(X):
    # TODO Fix Softmax Function
    #s = sum(list(map(lambda x: math.exp(x), X)))
    #func = np.vectorize(lambda x: math.exp(x) / s)
    return extmath.softmax(X)