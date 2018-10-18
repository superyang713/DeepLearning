"""
The simplest way to initialize weights is by using:
    np.random.randn() * 0.01
Where 0.01 is the scaling factor. However, this does not work very well.

He initialization is recommended for layers with a Relu activation. Another
popular initialization method is Xavier initialization. Both are very similar
except that they use different scaling factor:

    1. Scaling factor for He initialization:
        sqrt(1 / dimension of the previous layer)
    2. Scaling factor for He initialization:
        sqrt(2 / dimension of the previous layer)
"""


import numpy as np


def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dict containing your parameters "W1", "b1", ...,
        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
        b1 -- bias vector of shape (layers_dims[1], 1)
        ...
        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
        bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1

    for l in range(1, L + 1):
        scaling_factor = np.sqrt(2 / layers_dims[l - 1])
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]) * scaling_factor
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
