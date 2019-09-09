"""
This module contains all the building blocks to build a Recurrent Neural Network architecture

NOTE: the activation function implemented here is the hyperbolic tangent.
"""

import numpy as np
import nn_utils as utils


def rnn_cell_forward(xt, a_prev, parameters):
    """
    This function implements the computation of the forward pass for a single RNN cell

    Arguments:
    xt -- the input data at timestep 't', numpy array of shape (n_x, m)
    a_prev -- activation value at timestep 't-1', numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
        ba --  Bias, numpy array of shape (n_a, 1)
        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
        a_next -- next hidden state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cell_cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    # Retrieve parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # Compute activation state
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

    # Compute cell's output
    yt_pred = utils.softmax(np.dot(Wya, a_next) + by)

    # Store values for backpropagation
    cell_cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cell_cache


def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
        ba --  Bias numpy array of shape (n_a, 1)
        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    cache_list -- tuple of values needed for the backward pass, contains (list of caches, x)
    """

    # Initialize caches, a list that will contain the all the cache variable for all RNN cells
    cache_list = []

    # Compute dimensions of parameters
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # Initialize "a", "y_pred" and "a_next"
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])
    a_next = a0

    # Loop over all time-steps
    for t in range(T_x):
        # compute the state of the current RNN cell
        a_next, yt_pred, cell_cache = rnn_cell_forward(x[:,:,t], a_next, parameters)

        # update the hidden state value for the current time-step
        a[:, :, t] = a_next

        # save the prediction value in "y"
        y_pred[:, :, t] = yt_pred

        # save the cache for the current time-step
        cache_list.append(cell_cache)

    # Store values needed for backward propagation
    cache = (cache_list, x)

    return a, y_pred, cache


def rnn_cell_backpropagation(da_next, cell_cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
        dx -- Gradients of input data, of shape (n_x, m)
        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cell_cache

    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Way = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    

    pass


# TESTS
np.random.seed(1)
n_x = 5
m = 10
n_a = 8
n_y = 4

xt = np.random.randn(n_x, m)
a_prev = np.random.randn(n_a, m)
pars = {
    "Wax": np.random.randn(n_a, n_x),
    "Waa": np.random.randn(n_a, n_a),
    "Wya": np.random.randn(n_y, n_a),
    "ba": np.random.randn(n_a, 1),
    "by": np.random.randn(n_y, 1)
}

a, y_pred, cache = rnn_cell_forward(xt, a_prev, pars)
print(y_pred)
print(a.shape)
print(y_pred.shape)
print(cache.__len__())
