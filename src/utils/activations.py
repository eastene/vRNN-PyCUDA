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
    smax = np.empty_like(X)
    for i in range(X.shape[1]):
        exps = np.exp(X[:, i] - np.max(X[:, i]))
        smax[:,i] = exps / np.sum(exps)
    return smax