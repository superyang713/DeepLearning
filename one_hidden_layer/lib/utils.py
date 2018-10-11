"""
helper functions for One Layer Neural Network.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np


def reshape_X(X):
    """
    The original shape of X is (n_samples, n_features). In Neuro Network, X \
        should be reshaped to (n_features, n_samples)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1).T


def reshape_y(y):
    return y.reshape(1, -1)


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def load_planar_dataset():
    np.random.seed(1)
    n_samples = 400
    n_samples_per_class = int(n_samples / 2)
    n_rays = 4
    dim = 2
    X = np.zeros((n_samples, dim))
    Y = np.zeros(n_samples)

    for j in range(2):
        ix = range(n_samples_per_class * j, n_samples_per_class * (j + 1))
        theta = np.linspace(
            j * 3.12, (j + 1) * 3.12, n_samples_per_class
        ) + np.random.randn(n_samples_per_class) * 0.2
        radius = (
            n_rays * np.sin(4 * theta) +
            np.random.randn(n_samples_per_class) * 0.2
        )
        X[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        Y[ix] = j

    return X, Y
