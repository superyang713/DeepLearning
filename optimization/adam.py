"""
Adam is one of the most effective optimization algorithms for training neural
networks. It combines ideas from RMSProp and Momentum.

How does it work:
    1. It calculates an exponentially weighted average of past gradients, and
    stores it in variables v (before bias correction) and v_corrected (with
    bias correction).

    2. It calculates an exponentially weighted average of the squares of the
    past gradients, and stores it in variable s (before bias correction) and
    s_corrected (with bias correction).

    3. It updates parameters in a direction based on combining information from
    step 1 and step 2.
"""


import numpy as np

from testCases import initialize_adam_test_case
from testCases import update_parameters_with_adam_test_case


def initialize_adam(parameters):
    """
    initializes v and s as two python dictionaries with:
        - keys: "dW1", "db1", ..., "dWL", "dbL"
        - values: np.zeros of the same shape as the corresponding gradients

    Arguments:
    parameters -- python dictionary containing your parameters.
        parameters["W" + str(l)] = Wl
        parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted
        average of the gradient.
        v["dW" + str(l)] = ...
        v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted
        average of the squared gradient.
        s["dW" + str(l)] = ...
        s["db" + str(l)] = ...
    """

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


def update_parameters_with_adam(
    parameters, grads, v, s,
    beta1=0.9, beta2=0.999, epsilon=1e-8, t=2,
    learning_rate=0.01
):
    """
    Update parameters using Adam

    Arguments:
    ---------
    parameters -- python dictionary containing your parameters:
        parameters['W' + str(l)] = Wl
        parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
        grads['dW' + str(l)] = dWl
        grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dict
    s -- Adam variable, moving average of the squared gradient, python dict
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    -------
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dict
    s -- Adam variable, moving average of the squared gradient, python dict
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        # Moving average of the gradients
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + \
            (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + \
            (1 - beta1) * grads["db" + str(l + 1)]

        # Compute bias-corrected first moment estimate
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] /\
            (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] /\
            (1 - beta1 ** t)

        # Moving average of the squared gradients
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + \
            (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + \
            (1 - beta2) * np.square(grads["db" + str(l + 1)])

        # Compute bias-corrected second moment estimate
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] /\
            (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] /\
            (1 - beta2 ** t)

        # Update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - \
            learning_rate * v_corrected["dW" + str(l + 1)] / \
            (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - \
            learning_rate * v_corrected["db" + str(l + 1)] / \
            (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s


parameters = initialize_adam_test_case()
v, s = initialize_adam(parameters)
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
print("s[\"dW1\"] = " + str(s["dW1"]))
print("s[\"db1\"] = " + str(s["db1"]))
print("s[\"dW2\"] = " + str(s["dW2"]))
print("s[\"db2\"] = " + str(s["db2"]))


parameters, grads, v, s = update_parameters_with_adam_test_case()
parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
print("s[\"dW1\"] = " + str(s["dW1"]))
print("s[\"db1\"] = " + str(s["db1"]))
print("s[\"dW2\"] = " + str(s["dW2"]))
print("s[\"db2\"] = " + str(s["db2"]))
