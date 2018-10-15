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
