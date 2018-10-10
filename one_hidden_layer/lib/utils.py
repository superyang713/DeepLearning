import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


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
