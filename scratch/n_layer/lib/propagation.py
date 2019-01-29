"""
helper functions for N Layers Deep Neural Network.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np
from collections import namedtuple


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


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer

    Parameters:
    ----------
    dA : post-activation gradient for current layer l
    cache : namedtuple of (linear_cache, activation_cache) we store during \
        forward propagation for computing backward propagation efficiently
    activation : the activation to be used in this layer, stored as a text \
        string: "sigmoid" or "relu"

    Returns:
    -------
    dA_prev : Gradient of the cost with respect to the activation
    dW : Gradient of the cost with respect to W (current layer l)
    db : Gradient of the cost with respect to b (current layer l)
    """
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)

    return dA_pre, dW, db


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
    linear_cache = (A_pre, W, b)

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
    activation_cache = Z

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

    activation_cache = Z

    return A, activation_cache


def linear_backward(dZ, linear_cache):
    """
    Implement the linear portion of backward propagation.

    Parameters:
    ----------
    dZ : Gradient of the cost with respect to the linear output.
    cache : namedtuple of values (A_prev, W, b) from the forward propagation \
        in the current layer

    Returns:
    dA_prev : Gradient of the cost with respect to the activation (of the \
        previous layer l-1), same shape as A_prev
    dW : Gradient of the cost with respect to W (current layer l), same shape \
        as W
    db : Gradient of the cost with respect to b (current layer l), same shape \
        as b
    """
    A_pre, W, b = linear_cache
    n_samples = A_pre.shape[1]

    dW = 1 / n_samples * np.dot(dZ, A_pre.T)
    db = 1 / n_samples * np.sum(dZ, axis=1, keepdims=True)
    dA_pre = np.dot(W.T, dZ)

    assert (dA_pre.shape == A_pre.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_pre, dW, db


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

    Z = activation_cache

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

    Z = activation_cache

    dZ = np.array(dA, copy=True)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ
