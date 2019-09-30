"""
This module contains all the building blocks to build a Recurrent Neural Network architecture

NOTE: the activation function implemented here is the hyperbolic tangent.
"""

import numpy as np
from nn_architects import utilities as utils


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
    Implement the forward propagation of the recurrent neural network.

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

    # Computation of gradients with respect to the parameters
    # The general expressions to compute the derivatives of the cost function with respect to the parameters are
    # from the chain rule. These expressions are summarized as follows:
    # - dWax: dJ/dWax = dJ/da_next * da_next/dz * dz/dWax
    # - dWaa: dJ/dWaa = dJ/da_next * da_next/dz * dz/dWaa
    # - dba: dJ/db = dJ/da_next * da_next/dz * dz/dba
    # where:
    # a = tanh(z), da = (1 - tanh(z)**2)*dz
    # z = Wax*x + Waa*a_prev + ba

    dz = (1 - a_next ** 2) * da_next
    dxt = np.dot(Wax.T, dz)
    dWax = np.dot(dz, xt.T)
    da_prev = np.dot(Waa.T, dz)
    dWaa = np.dot(dz, a_prev.T)
    dba = np.sum(dz, keepdims=True, axis=-1)

    # Store gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients

def rnn_backpropagation(da, cache):
    """
    Implementation of the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)

    Returns:
    gradients -- python dictionary containing:
        dx -- Gradient with respect to the input data, numpy-array of shape (n_x, m, T_x)
        da0 -- Gradient with respect to the initial hidden state, numpy-array of shape (n_a, m)
        dWax -- Gradient with respect to the input's weight matrix, numpy-array of shape (n_a, n_x)
        dWaa -- Gradient with respect to the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
    """

    # Retrieve cache
    (cache_t, x) = cache

    # Get cache values for the first time step
    (a1, a0, x1, parameters) = cache_t[0]

    # Compute variable dimensions
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # Initialize gradients
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prev_t = np.zeros((n_a, m))

    # Loop over all time-steps
    for t in reversed((range(T_x))):
        # Compute gradients for time step t
        gradients = rnn_cell_backpropagation(da[:, :, t] + da_prev_t, cache_t[t])
        dxt, da_prev_t, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], \
                                            gradients["dWaa"], gradients["dba"]

        # Increment global derivatives with respect to parameters by adding their derivative at time-step t
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    # Set da0 to the gradient of a which has been backpropagated through all time-steps
    da0 = da_prev_t

    # Store gradients in a dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients
