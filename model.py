"""
This module builds a NN model with the specified configuration options
"""

import numpy as np
import tensorflow as tf
from tensorflow.nn import rnn_cell


class Model:

    def __init__(self, options):
        # Unpack options
        self.n_hidden = options["n_hidden"]
        self.activation = options["activation"]


    def build_rnn(self, type="basic"):
        """
        Arguments:
             type -- a string defining the RNN type. It assumes three types: 'basic', 'gru' and 'lstm'
        """
        # Initialize a basic rnn_cell
        if type == "basic":
            self.rnn_cell = rnn_cell.BasicRNNCell(self.n_hidden, self.activation)
        elif type == "gru":
            raise ValueError("To be implemented! Choose another RNN type")
        elif type == "lstm":
            raise ValueError("To be implemented! Choose another RNN type")
        else:
            raise ValueError("Choose a valid RNN type: basic, gru or lstm")

        return self.rnn_cell


    def optimize(self, data):
        """
        This function optimizes the model
        Arguments:
            data -- list containing the inputs and label data, e.g. [X, Y]
        """
        # Get data shape
        n_inputs, n_examples, Tx_max = data[0].shape

        # Initialize data placeholders for the input data (X) and label data (Y)
        X = tf.placeholder(tf.float64, shape=(n_inputs, n_examples, Tx_max), name="inputs")
        Y = tf.placeholder(tf.float64, shape=(n_inputs, n_examples, Tx_max), name="labels")
        print(X)
        print(Y)


if __name__ == "__main__":
    # Tests
    print("Running tests")
    print("Tensorflow version {}".format(tf.__version__))
    options = {
        "n_inputs": 30,
        "n_examples": 1000,
        "Tx_max": 50,
        "n_hidden": 64,
        "activation": "tanh"
    }
    # Generate data
    X, Y = (np.random.randn(30, 1000, 50), np.random.randn(30, 1000, 50))

    mod = Model(options)
    rnn_cell = mod.build_rnn()
    mod.optimize((X, Y))
