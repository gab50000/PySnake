import numpy as np


def cross_entropy(y, y_net):
    n = y.shape[0]
    return -1 / n * (y * np.log(y_net) + (1 - y) * np.log(1 - y_net)).sum(axis=0)


def cross_entropy_deriv(y, y_net):
    n = y.shape[0]
    return -1 / n * (np.log(y_net) - np.log(1 - y_net))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def q_learning_update(q_old, q_next_estimate, reward, discount, learning_rate):
    return q_old + learning_rate * (reward + discount * q_next_estimate - q_old)


class FFN:
    def __init__(self, in_size, hl_size, out_size, dna=None):
        if dna:
            self.dna = dna
        else:
            self.dna = np.random.randn((in_size + 1) * hl_size + (hl_size + 1) * out_size)
        self.W1 = self.dna[:(in_size + 1) * hl_size].reshape((in_size + 1, hl_size))
        self.W1 /= self.W1.size
        self.W2 = self.dna[(in_size + 1) * hl_size:].reshape((hl_size + 1, out_size))
        self.W2 /= self.W2.size

    def prop(self, x1):
        x2 = np.maximum(0, x1 @ self.W1[:-1] + self.W1[-1])
        x3 = x2 @ self.W2[:-1] + self.W2[-1]
        softmax_x3 = np.exp(x3 - x3.max(axis=-1))
        softmax_x3 /= softmax_x3.sum(axis=-1)
        return softmax_x3

    def backprop(self, training_input, training_output, gamma=1.0, verbose=False):
        """Do backpropagation by calculating the gradient of the loss function
        with respect to the weights W1 and W2.
        Each variable dX calculates the derivative dloss/dX"""

        # Forward prop
        x2 = np.maximum(0, training_input @ self.W1[:-1] + self.W1[-1])
        x3 = x2 @ self.W2[:-1] + self.W2[-1]
        exp_scores = np.exp(x3 - x3.max(axis=-1))
        probs = exp_scores / exp_scores.sum(axis=-1)
        # As training_output holds the correct action, check how high the
        # probability assigned by softmax is for it
        # If it is close to one, loss -> 0
        # If it is close to zero, loss -> ∞
        loss = -np.log(probs[range(training_input.shape[0]), training_output])

        dW1 = np.empty((training_input.shape[0], *self.W1.shape))
        dW2 = np.empty((training_input.shape[0], *self.W2.shape))

        # Backpropagation
        # dL/dxk = probs(xi) - delta_ik
        # -> the gradient for the correct output is the difference between 1
        # and its assigned probability
        # The gradient for the wrong outputs is the difference between 0 and
        # their assigned probabilities

        dx3 = loss
        dx3[range(training_input.shape[0]), training_output] -= 1
        # W2 has shape (x2.shape[1] + 1, x3.shape[1])
        dW2[:, :-1] = x2[:, :, None] * dx3[:, None, :]
        dW2[:, -1] = dx3
        # Do matrix multiplication for each training value
        dx2 = np.einsum("jk, ik -> ij", self.W2[: -1], dx3)
        dx2[x2 == 0] = 0
        # W1 has shape (x_train.shape[1], x2.shape[1])
        dW1[:, :-1] = training_input[:, :, None] * dx2[:, None, :]
        dW1[:, -1] = dx2

        self.W1 += -gamma * dW1.sum(axis=0)
        self.W2 += -gamma * dW2.sum(axis=0)

        return loss.mean()
