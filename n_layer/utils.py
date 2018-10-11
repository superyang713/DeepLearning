"""
helper functions for N Layers Deep Neural Network.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Parameters:
    ----------
    Z : numpy array of any shape

    Returns:
    -------
    A : output of sigmoid(z), same shape as Z
    cache : returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Parameters:
    ----------
    dA : post-activation gradient, of any shape
    cache : 'Z' where we store for computing backward propagation efficiently

    Returns:
    -------
    dZ : Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu(Z):
    """
    Implement the RELU function.

    Parameters:
    ----------
    Z : Output of the linear layer, of any shape

    Returns:
    -------
    A : Post-activation parameter, of the same shape as Z
    cache : dict containing "A" ;
        Stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)

    cache = Z

    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Parameters:
    ----------
    dA : post-activation gradient, of any shape
    cache : 'Z' where we store for computing backward propagation efficiently

    Returns:
    -------
    dZ : Gradient of the cost with respect to Z
    """

    Z = cache

    # just converting dz to a correct object.
    dZ = np.array(dA, copy=True)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ
