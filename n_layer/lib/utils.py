"""
helper functions for One Layer Neural Network.
"""

# Author: Yang Dai <daiy@mit.edu>


import h5py
import numpy as np
import matplotlib.pyplot as plt


def reshape_X(X):
    """
    The original shape of X is (n_samples, n_features). In Neuro Network, X \
        should be reshaped to (n_features, n_samples)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1).T


def reshape_y(y):
    return y.reshape(1, -1)


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


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(test_set_y_orig.shape[0])

    return (train_set_x_orig,
            train_set_y_orig,
            test_set_x_orig,
            test_set_y_orig,
            classes)


def plot_image(X, y, index):
    """
    Visualize an example image in the dataset.
    """
    plt.imshow(X[index])
    plt.axis('off')
    plt.show()
