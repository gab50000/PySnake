import pickle

import numpy as np


def cross_entropy(y, y_net):
    n = y.shape[0]
    return -1 / n * (y * np.log(y_net) + (1 - y) * np.log(1 - y_net)).sum(axis=0)


def sigmoid(x):
    ex = np.exp(x)
    return ex / (1 + ex)


class NeuralNet:
    def __init__(self, in_size, hl_size, out_size, dna=None):
        if dna is not None:
            self.dna = dna
        else:
            size = (in_size + 1) * hl_size + (hl_size + 1) * out_size
            self.dna = np.random.randn(size)
        self.W1 = self.dna[: (in_size + 1) * hl_size].reshape((in_size + 1, hl_size))
        self.W2 = self.dna[(in_size + 1) * hl_size :].reshape((hl_size + 1, out_size))

        if dna is None:
            self.W1 /= np.sqrt(self.W1.shape[0])
            self.W2 /= np.sqrt(self.W2.shape[0])

    def forward(self, x1):
        x2 = np.tanh(x1 @ self.W1[:-1] + self.W1[-1])
        x3 = x2 @ self.W2[:-1] + self.W2[-1]
        softmax_x3 = np.exp(x3 - x3.max(axis=-1, keepdims=True))
        softmax_x3 /= softmax_x3.sum(axis=-1, keepdims=True)
        return softmax_x3

    def decide(self, x1):
        return np.argmax(self.forward(x1))

    def __getstate__(self):
        return {"W1": self.W1, "W2": self.W2}

    def __setstate__(self, state):
        self.__dict__.update(state)
