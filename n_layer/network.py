"""
Neuro Network Classifier with One Hidden Layer.
The structure is inspired by scikit-learn python library.

Important notes:
    1. X shape from the data source is (n_samples, n_features)
    2. y shape from the data source is (n_samples,)
    3. Internally, X is converted to (n_features, n_samples), and y is \
        converted to (1, n_samples)
    4. y_predict shape is (n_samples,)
    5. W shape is (n_neurons, n_features)
    6. b is (n_neurons, 1)
    7. Therefore, internally,  Z = WX + b
    9. Optimization is done by batch gradient descent.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np
import matplotlib.pyplot as plt

from propagation import linear_activation


class NeuralNetworkClassifier:
    def __init__(self, layer_dims):
        """
        Parameters:
        ----------
        layer_dims : python array (list)
            the dimensions of each layer in our network

        L :  Size of the neural network
        caches : list of caches containing
            every cache of linear_relu_forward, range(1, L -1)
            the cache of linear_sigmoid_forward, size 1.
        params : dict containing all parameters in the network.
        """
        self.layer_dims = layer_dims

        self.L = len(self.layer_dims)

        self.params = {}
        self.caches = []

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _init_params(self):
        """
        Initialize all the parameters in the Neural Network.
        """

        np.random.seed(3)
        for l in range(1, self.L):
            self.params['W' + str(l)] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]
            ) * 0.01

            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert(self.params['W' + str(l)].shape == (
                self.layer_dims[l], self.layer_dims[l - 1]
            ))
            assert(self.params['b' + str(l)].shape == (self.layer_dims[l], 1))

    def _forward_propagate(self, X):
        """
        Implement forward propagation for the [LINEAR -> REUL] * (L -1) -> \
            LINEAR -> SIGMOID computation.

        Parameters:
        ----------
        X : input data of size (n_features, n_samples)

        Returns:
        -------
        y_hat : The output of the neural network, which is just the output \
            of the final activation function.
        """

        A_pre = X
        for l in range(1, self.L):
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]
            A_pre, cache = linear_activation(A_pre, W, b, "relu")
            self.caches.append(cache)

        W = self.params["W" + str(self.L)]
        b = self.params["b" + str(self.L)]
        y_hat, cache = linear_activation(A_pre, W, b, "sigmoid")
        self.caches.append(cache)
