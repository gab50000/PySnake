import unittest

import numpy as np

from snake.q_learning import FFN


class TestNeuralNet(unittest.TestCase):
    ffn = FFN(in_size=10, hl_size=10, out_size=4)

    training_size = 10000
    training_input = np.random.uniform(-2, 2, size=(training_size, 10))
    training_output = np.zeros((training_size, 4))
    #random_indices = np.random.randint(0, 4, size=training_size)
    #training_output[range(training_size), random_indices] = 1
    training_output[:, 0] = 1

    ffn.backprop(training_input, training_output)
