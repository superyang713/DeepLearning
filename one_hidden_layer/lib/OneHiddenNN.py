"""
Neuro Network Classifier with One Hidden Layer.
The structure is inspired by scikit-learn python library.

Important notes:
    1. Activation function in the hidden layer is tanh.
    2. Activation function in the output layer is sigmoid, so the output \
        layer size is 1. It can be greater than one if is multiclass \
        classification and use activation function such as softmax.
    3. X shape from the data source is (n_samples, n_features)
    4. y shape from the data source is (n_samples,)
    5. Internally, X is converted to (n_features, n_samples), and y is \
        converted to (1, n_samples)
    6. y_predict shape is (n_samples,)
    7. W shape is (n_neurons, n_features)
    8. b is (n_neurons, 1)
    9. Therefore, internally,  Z = WX + b
    10. Optimization is done by batch gradient descent.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np
import matplotlib.pyplot as plt
from .utils import reshape_X, reshape_y, sigmoid


class OneHiddenLayerClassifier:
    def __init__(self, n_hidden1=5, learning_rate=0.005, max_iter=2000):
        # Hyperparameter
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # Number of neurons in the hidden layer
        self.n_hidden1 = n_hidden1

        # Parameters
        self.params = {}

        self.cache = {}
        self.grads = {}
        self.costs = []

    def fit(self, X, y):
        X = reshape_X(X)
        y = reshape_y(y)
        n_features, n_samples = X.shape
        n_output = y.shape[0]
        self._init_params(n_features, n_output)

        # Gradient Descent
        for i in range(self.max_iter):
            output = self._forward_propagate(X)
            cost = self._compute_cost(output, y)
            self.costs.append(cost)
            self._backward_progagate(X, y)
            self._update_parameters()

    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Parameters:
        ----------
        self.params : dict containing all parameters in the network.
        X : input data of size (n_samples, n_features)

        Returns
        y_predict -- vector of predictions of our model (n_samples,)
        """
        X = reshape_X(X)
        output = self._forward_propagate(X)
        output = output.ravel()

        y_predict = np.rint(output)

        return y_predict

    def _init_params(self, n_features, n_output):
        """
        Parameters:
        ----------
        n_features : size of the input layer
        n_output : size of the output layer
        self.n_hidden1 : size of the hidden layer

        Returns:
        -------
        self.params : dict containing parameters:
            W1 -- weight matrix of shape (self.n_hidden1, n_features)
            b1 -- bias vector of shape (self.n_hidden1, 1)
            W2 -- weight matrix of shape (n_output, n_hidden1)
            b2 -- bias vector of shape (self.n_output, 1)
        """

        W1 = np.random.randn(self.n_hidden1, n_features) * 0.01
        b1 = np.zeros((self.n_hidden1, 1))
        W2 = np.random.randn(n_output, self.n_hidden1) * 0.01
        b2 = np.zeros((n_output, 1))

        assert(W1.shape == (self.n_hidden1, n_features))
        assert(b1.shape == (self.n_hidden1, 1))
        assert(W2.shape == (n_output, self.n_hidden1))
        assert(b2.shape == (n_output, 1))

        self.params = {
            'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2
        }

    def _forward_propagate(self, X):
        """
        Parameters:
        ----------
        X : input data of size (n_features, n_samples)
        self.params : dict containing all parameters in the network.

        Returns:
        -------
        output : The sigmoid output of the second activation, although in \
            this case is A2.
        cache : dict containing "Z1", "A1", "Z2" and "A2"
        """
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        output = A2

        assert(A2.shape == (1, X.shape[1]))

        self.cache = {
            'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2
        }

        return output

    @staticmethod
    def _compute_cost(output, y):
        """
        Computes the cross-entropy cost for logistic regression.

        Parameters:
        ----------
        output : The sigmoid output of the second activation, although in \
            this case is A2.
        y : true labels vector of shape (1, n_samples)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        n_samples = y.shape[1]
        cost = -1 / n_samples * (
            np.dot(y, np.log(output.T)) + np.dot(1 - y, np.log(1 - output.T))
        )

        cost = np.squeeze(cost).item()

        assert(isinstance(cost, float))

        return cost

    def _backward_progagate(self, X, y):
        """
        Implement the backward propagation.

        Parameters:
        ----------
        self.params : dict containing our parameters
        self.cache :  dict containing "Z1", "A1", "Z2" and "A2".
        X : input data of size (n_features, n_samples)
        y : true labels vector of shape (1, n_samples)

        Returns:
        self.grads : dict containing gradients with respect to parameters
        """
        n_samples = X.shape[1]

        # Parameters
        W2 = self.params['W2']

        # Cache
        A1 = self.cache['A1']
        A2 = self.cache['A2']

        # Calculate gradients using chain rule.
        dZ2 = A2 - y
        dW2 = 1 / n_samples * np.dot(dZ2, A1.T)
        db2 = 1 / n_samples * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = 1 / n_samples * np.dot(dZ1, X.T)
        db1 = 1 / n_samples * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {
            'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2
        }

    def _update_parameters(self):
        """
        Updates parameters using the gradient descent update rule.
        """
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]

        dW1 = self.grads["dW1"]
        db1 = self.grads["db1"]
        dW2 = self.grads["dW2"]
        db2 = self.grads["db2"]

        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2

        self.params = {
            "W1": W1, "b1": b1, "W2": W2, "b2": b2
        }

    def get_accuracy(self, X, y):
        y_predict = self.predict(X)
        n_samples = X.shape[0]
        accuracy = np.sum(y_predict == y) / n_samples
        return accuracy

    def plot_decision_boundary(self, X, y):
        """
        Make a decision boundary plot for the trained Decision Tree \
            Classifier and examine how trained or test sets fit into the plot.

        Parameter:
        ---------
        X : numpy array
            The shape must be (n_samples, 2), which means there can only be \
                two features.
        y: numpy array
            Labels. 1-D vector.
        """

        # Plot decision boundary.
        X1_coor, X2_coor = np.meshgrid(
            np.arange(start=X[:, 0].min()-1, stop=X[:, 0].max()+1, step=0.01),
            np.arange(start=X[:, 1].min()-1, stop=X[:, 1].max()+1, step=0.01)
        )
        X_grid = np.array([X1_coor.ravel(), X2_coor.ravel()]).T
        Y_grid = self.predict(X_grid).reshape(X1_coor.shape)
        cmap = plt.cm.Spectral

        plt.contourf(X1_coor, X2_coor, Y_grid, alpha=0.75, cmap=cmap)
        plt.xlim(X1_coor.min(), X1_coor.max())
        plt.ylim(X2_coor.min(), X2_coor.max())

        # Plot datapoints in the decision boundary graph.
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
