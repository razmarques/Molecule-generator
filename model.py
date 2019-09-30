"""
This module contains functions to build a many to many Recurrent Neural Network architecture
"""

from nn_architects import rnn_simple
import numpy as np


def build_input_data(smiles_df, characters_dict):

    eos_token = characters_dict[0]
    n_x = int(characters_dict.__len__() / 2)
    m = smiles_df.shape[0]
    Tx_max = max([len(ch) for ch in smiles_df["smiles"]]) + 1 # added one unit to accommodate the EOS token

    # Print data statistics
    print("Number of training examples: {0}".format(m))
    print("Number of SMILES characters: {0}".format(n_x))
    print("Maximum SMILES length: {0}".format(Tx_max))

    # Initialize data array
    X = np.zeros((n_x, m, Tx_max))
    Y = np.zeros((n_x, m, Tx_max)) # the Y matrix is the data in X shifted on time-step further since we want the RNN to predict the next character in the sequence.

    # Fill X with one-hot vectors for each SMILES training example
    # NOTE: the x one-hot vector for the first time-step is set to the zero.
    for iex in range(m):
        # Extract and canonize the ith SMILES
        smi = smiles_df.loc[iex, "smiles"]
        # smi = Chem.CanonSmiles(smiles_df.loc[iex, "smiles"] + eos_token) # This takes too long. Disabled for the time being

        for t in range(len(smi)):
            ichar = characters_dict[smi[t]]
            X[ichar, iex, t+1] = 1
            Y[ichar, iex, t] = 1

            # If this is the last character, add the EOS character to t+1
            if t == len(smi):
                ieos = characters_dict[eos_token]
                Y[ieos, iex, t+1] = 1

    return X, Y


def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values

    Returns:
    parameters -- python dictionary containing:
        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
        b --  Bias, numpy array of shape (n_a, 1)
        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """

    Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    ba = np.zeros((n_a, 1))  # hidden bias
    by = np.zeros((n_y, 1))  # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

    return parameters


def mini_batch_partition():
    # TODO
    pass


def optimize(X, Y, n_hidden, learning_rate, n_iter):
    """
    This function executes one iteration for the model optimization

    Inputs:
        X -- Input data for every time-step with shape (n_x, m, T_x)
        Y -- The same as X but with a time step index shifted to the left.
        n_hidden -- number of hidden nodes per RNN cell.
        learning_rate -- learning rate for the model.
        niter -- number of iterations

    Outputs:
        loss -- value of the loss function (cross-entropy)
        gradients -- python dictionary containing:
            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
            dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
            db -- Gradients of bias vector, of shape (n_a, 1)
            dby -- Gradients of output bias vector, of shape (n_y, 1)
    """

    # Compute data sizes
    n_x, n_y = (X.shape[0], X.shape[0])
    m = X.shape[1]

    # Initialize parameters
    parameters = initialize_parameters(n_hidden, n_x, n_y)

    # Initialize a_prev
    a_init = np.zeros((n_hidden, m))

    for iter in range(n_iter):

        # Compute forward propagation
        a, y_pred, cache = rnn_simple.rnn_forward(X, a_init, parameters)


    return loss, gradients

def sample():
    # TODO
    pass
