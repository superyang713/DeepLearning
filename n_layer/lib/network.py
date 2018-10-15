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

from .propagation import linear_activation, linear_activation_backward
from .utils import reshape_X, reshape_y


class NeuralNetworkClassifier:
    def __init__(self, layer_dims, learning_rate=0.1, max_iter=3000):
        """
        Parameters:
        ----------
        learning_rate: float
        layer_dims : python array (list)
            the dimensions of each layer in our network
        max_iter: int
            number of iterations of the optimization loop.

        L :  Size of the neural network
        caches : list of caches containing
            every cache of linear_relu_forward, range(1, L -1)
            the cache of linear_sigmoid_forward, size 1.
        params : dict containing all parameters in the network.
        grads : dict with the gradients
            grads["dA" + str(l)] = ...
            grads["dW" + str(l)] = ...
            grads["db" + str(l)] = ...
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.L = len(self.layer_dims)

        self.costs = []
        self.caches = []
        self.params = {}
        self.grads = {}

    def fit(self, X, y):
        """
        Train the model.

        Parameters:
        ----------
        X : input data of size (n_samples, n_features)
        y : true labels vector of shape (n_samples)

        Returns:
        -------
        """
        np.random.seed(1)

        X = reshape_X(X)
        y = reshape_y(y)

        self._init_params()

        for i in range(self.max_iter):
            AL = self._forward_propagate(X)
            cost = self._compute_cost(AL, y)
            self.costs.append(cost)
            self._backward_progagate(AL, y)
            self._update_parameters()

    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Parameters:
        ----------
        X : input data of size (n_samples, n_features)

        Returns
        y_predict -- vector of predictions of our model (n_samples,)
        """
        X = reshape_X(X)
        AL = self._forward_propagate(X)
        AL = AL.ravel()

        y_predict = np.rint(AL)

        return y_predict

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
        AL : The output of the neural network (y_hat), which is the output \
            of the final activation function.
        """

        A_pre = X
        for l in range(1, self.L - 1):
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]
            A_pre, cache = linear_activation(A_pre, W, b, "relu")
            self.caches.append(cache)

        W = self.params["W" + str(self.L - 1)]
        b = self.params["b" + str(self.L - 1)]
        AL, cache = linear_activation(A_pre, W, b, "sigmoid")
        self.caches.append(cache)

        assert(AL.shape == (1, X.shape[1]))

        return AL

    @staticmethod
    def _compute_cost(AL, y):
        """
        Computes the cross-entropy cost.

        Parameters:
        ----------
        AL : numpy array of shape (1, n_samples)
            probability vector corresponding to the label predictions.
        y : true labels vector of shape (1, n_samples)
        self.params : dict containing all parameters in the network.

        Returns:
        cost : cross-entropy cost
        """
        n_samples = y.shape[1]
        cost = -1 / n_samples * (
            np.dot(y, np.log(AL.T)) + np.dot(1 - y, np.log(1 - AL.T))
        )

        cost = np.squeeze(cost).item()

        assert(isinstance(cost, float))

        return cost

    @staticmethod
    def _compute_dAL(AL, y):
        """
        Compute the derivative of cost with respect to AL.

        Parameters:
        ----------
        AL : numpy array of shape (1, n_samples)
            probability vector corresponding to the label predictions.
        y : true labels vector of shape (1, n_samples)
        self.params : dict containing all parameters in the network.

        Returns:
        dAL : numpy array of the same shape as y.
            This post-activation gradient dAL is used to initiate back \
            propagation.
        """

        dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
        return dAL

    def _backward_progagate(self, AL, y):
        """
        Implement the backward propagation.

        Parameters:
        ----------
        AL : numpy array of shape (1, n_samples)
            probability vector corresponding to the label predictions.
        y : true labels vector of shape (1, n_samples)

        Returns:
        -------
        """
        dAL = self._compute_dAL(AL, y)

        sigmoid_cache = self.caches[-1]

        (self.grads["dA" + str(self.L - 1)],
         self.grads["dW" + str(self.L - 1)],
         self.grads["db" + str(self.L - 1)]) = linear_activation_backward(
             dAL, sigmoid_cache, "sigmoid"
         )

        for l in reversed(range(self.L - 2)):
            current_cache = self.caches[l]

            (self.grads["dA" + str(l + 1)],
             self.grads["dW" + str(l + 1)],
             self.grads["db" + str(l + 1)]) = linear_activation_backward(
                self.grads['dA' + str(l + 2)], current_cache, "relu"
            )

    def _update_parameters(self):
        """
        Updates parameters using the gradient descent update rule.
        """
        for l in range(1, self.L):
            self.params['W' + str(l)] = self.params['W' + str(l)] - \
                self.learning_rate * self.grads['dW' + str(l)]
            self.params['b' + str(l)] = self.params['b' + str(l)] - \
                self.learning_rate * self.grads['db' + str(l)]

    def get_accuracy(self, X, y):
        y_predict = self.predict(X)
        n_samples = X.shape[0]
        accuracy = np.sum(y_predict == y) / n_samples
        return accuracy
