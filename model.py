"""
This module builds a NN model with the specified configuration options
"""

import numpy as np
import tensorflow as tf


class Model:

    def __init__(self, options):
        # Define instance variables
        self.options = options

        # Get data shape
        data_shape = (options["n_inputs"], options["n_examples"], options["Tx_max"])

        # Initialize data placeholders for the input data (X) and label data (Y)
        self.X = tf.placeholder(tf.float64, shape=data_shape, name="inputs")
        self.Y = tf.placeholder(tf.float64, shape=data_shape, name="labels")


    def build_rnn(self, type):
        """
        Arguments:
             type -- a string defining the RNN type. It assumes three types: 'simple', 'gru' and 'lstm'
        """
        #

        rnn_cell = tf.nn.rnn_cell.RNNCell()
        pass


    def initialize_parameters(self):
        # TODO
        pass


    def optimize(self):
        pass


if __name__ == "__main__":
    # Tests
    print("Running tests")
    print("Tensorflow version {}".format(tf.__version__))
    configs = {}
