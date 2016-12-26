import numpy as np


def cross_entropy(x, y, y_net):
    n = x.shape[0]
    return -1 / n * (y * np.log(y_net) + (1 - y) * np.log(1 - y_net))


class SnakeNet:

    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.standard_normal((input_size, hidden_size))
        self.W1 /= np.sqrt(self.W1.size)
        self.W2 = np.random.standard_normal((hidden_size, 4))
        self.W2 /= np.sqrt(self.W2.size)

    def forwardprob(self, x1):
        x2 = np.maximum(0, self.W1 @ x1)
        y = np.maximum(0, self.W2 @ x2)

        # Calculate softmax
        y -= y.max()  # Improves numerical stability
        out = -y + np.log((np.exp(y)).sum())

        return out

    def backprop(self):
        pass


class FFN:
    def __init__(self, in_size, hl_size, out_size):
        self.W1 = np.random.randn(in_size + 1, hl_size)
        self.W1 /= self.W1.size
        self.W2 = np.random.randn(hl_size + 1, out_size)
        self.W2 /= self.W2.size

    def prop(self, x1):
        x2 = np.maximum(0, x1 @ self.W1[:-1] + self.W1[-1])
        x3 = x2 @ self.W2[:-1] + self.W2[-1]
        softmax_x3 = np.exp(x3-x3.max(axis=-1))
        softmax_x3 /= softmax_x3.sum(axis=-1)
        return softmax_x3

    def backprop(self, x_train, y_train, gamma=1.0, verbose=False):
        """Do backpropagation by calculating the gradient of the loss function
        with respect to the weights W1 and W2."""
        # Forward prop
        x2 = np.maximum(0, x_train @ self.W1[:-1] + self.W1[-1])
        x3 = x2 @ self.W2[:-1] + self.W2[-1]
        softmax_x3 = np.exp(x3-x3.max(axis=-1))
        softmax_x3 /= softmax_x3.sum(axis=-1)
        loss = cross_entropy(x_train, y_train, softmax_x3)

        dW1 = np.empty((x_train.shape[0], *self.W1.shape))
        dW2 = np.empty((x_train.shape[0], *self.W2.shape))

        # Backpropagation
        dx3 = x3 - y_train
        # W2 has shape (x2.shape[1] + 1, x3.shape[1])
        dW2[:, :-1] = x2[:, :, None] * dx3[:, None, :]
        dW2[:, -1] = dx3
        # import ipdb; ipdb.set_trace()
        # Do matrix multiplication for each training value
        dx2 = np.einsum("jk, ik -> ij", self.W2[: -1], dx3)
        dx2 *= sigmoid_deriv(x2)
        # W1 has shape (x_train.shape[1], x2.shape[1])
        dW1[:, :-1] = x_train[:, :, None] * dx2[:, None, :]
        dW1[:, -1] = dx2

        self.W1 += -gamma * dW1.sum(axis=0)
        self.W2 += -gamma * dW2.sum(axis=0)

        return loss.mean()
