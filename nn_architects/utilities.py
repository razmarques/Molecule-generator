"""
Utility functions for the implementation of Neural Network architectures
"""

import numpy as np


def clip(gradients, maxValue):
    """
    Clips gradients between minimum and maximum
    """

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


def loss_function(y_hat, y):
    """
     Computes the cross-entropy loss function for the SoftMax layer
     Arguments:
         y_hat -- Tensor of output predictions (n_y, m, T_x)
         y -- Tensor of output labels (n_y, m, T_x)
     Returns:
         loss -- array of loss function values for each time-step
    """

    m = y_hat.shape[1]
    T_x = y_hat.shape[2]

    # Initialize loss array
    loss = np.zeros((1, T_x))

    for t in range(T_x):
        yt_hat = y_hat[:, :, t]
        yt = y[:, :, t]

        loss[t] = (1/m)*np.sum(- np.sum(yt*np.log(yt_hat), axis=0))

    return loss


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
