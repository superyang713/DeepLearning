"""
helper functions for N Layers Deep Neural Network.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np
from collections import namedtuple


Linear_cache = namedtuple('Linear_cache', 'A_pre W b')
Activation_cache = namedtuple('Activation_cache', 'Z')
Cache = namedtuple('Cache', 'linear_cache activation_cache')


def linear_activation(A_pre, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Parameters:
    ----------
    A_pre : numpy array of shape (size of previous layer, n_samples)
        activations from previous layer (or input data).
    W : numpy array of shape (size of current layer, size of previous layer)
        weights matrix
    b : numpy array of shape (size of the current layer, 1)
        bias vector
    activation : text string: "sigmoid" or "relu"
        the activation to be used in this layer

    Returns:
    -------
    A : numpy array of shape (size of the current layer, n_samples)
        the output of the activation function, aka: post-activation value
    cache : dict containing "linear_cache" and "activation_cache";
        stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        Z, linear_cache = linear(A_pre, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear(A_pre, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_pre.shape[1]))

    cache = Cache(linear_cache, activation_cache)
    return A, cache


def linear(A_pre, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Parameters:
    ----------
    A_pre : numpy array of shape (size of previous layer, n_samples)
        activations from previous layer (or input data)
    W : numpy array of shape (size of current layer, size of previous layer)
        weights matrix
    b : numpy array of shape (size of the current layer, 1)
        bias vector

    Returns:
    -------
    cache : returns Z as well, useful during backpropagation
    Z : numpy array of shape (size of the current layer, n_samples)
        the input of the activation function, aka:pre-activation parameter
    cache : dict containing "A", "W" and "b"
        stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A_pre) + b
    assert(Z.shape == (W.shape[0], A_pre.shape[1]))
    linear_cache = Linear_cache(A_pre, W, b)

    return Z, linear_cache


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
    activation_cache = Activation_cache(Z)

    return A, activation_cache


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

    activation_cache = Activation_cache(Z)

    return A, activation_cache


def sigmoid_backward(dA, activation_cache):
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

    Z = activation_cache.Z

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu_backward(dA, activation_cache):
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

    Z = activation_cache.Z

    # just converting dz to a correct object.
    dZ = np.array(dA, copy=True)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ
