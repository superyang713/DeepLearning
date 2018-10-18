"""
Mini-batch gradient descent makes a parameter update after seeing just a subset
of examples, the direction of the update has some variabce, and so the path by
mini-batch gradient descent will still "oscillate" toward convergence. Using
momentum can reduce these oscillations.

Momentum takes into account the past gradients to smooth out the update. We
will store the 'direction' of the previous gradients in the variable v.
Formally, this will be the exponentially weighted average of the gradient on
previous steps. You can also think of v as the "velocity" of a ball rolling
downhill, building up speed (and momentum) according to the direction of the
gradient/slop of the hill.

To apply momentum to the gradient descent, the velocity should be first
initialized. During back propagation, the parameters will not be updated with
gradients, instead, they will be updated with the velocity.
"""


import numpy as np

from testCases import initialize_velocity_test_case
from testCases import update_parameters_with_momentum_test_case


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
        - keys: "dW1", "db1", ..., "dWL", "dbL"
        - values: np.zeros of the same shape as the corresponding gradients.
    Arguments:
    parameters -- python dictionary containing your parameters.
        parameters['W' + str(l)] = Wl
        parameters['b' + str(l)] = bl

    Returns:
    velocity -- python dictionary containing the current velocity.
        v['dW' + str(l)] = velocity of dWl
        v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2
    velocity = {}

    for l in range(L):
        velocity["dW" + str(l + 1)] = np.zeros(
            parameters["W" + str(l + 1)].shape
        )
        velocity["db" + str(l + 1)] = np.zeros(
            parameters["b" + str(l + 1)].shape
        )

    return velocity


def update_parameters_with_momentum(
    parameters, grads, velocity,
    beta=0.9,
    learning_rate=0.01
):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
        parameters['W' + str(l)] = Wl
        parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
        grads['dW' + str(l)] = dWl
        grads['db' + str(l)] = dbl
    velocity -- python dictionary containing the current velocity:
        velocity['dW' + str(l)] = ...
        velocity['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    velocity -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2

    for l in range(L):
        # Compute velocity
        velocity["dW" + str(l + 1)] = beta * velocity["dW" + str(l + 1)] + \
            (1 - beta) * grads["dW" + str(l + 1)]
        velocity["db" + str(l + 1)] = beta * velocity["db" + str(l + 1)] + \
            (1 - beta) * grads["db" + str(l + 1)]

        # Update parameters
        parameters["dW" + str(l + 1)] = parameters["W" + str(l + 1)] - \
            learning_rate * velocity["dW" + str(l + 1)]
        parameters["db" + str(l + 1)] = parameters["b" + str(l + 1)] - \
            learning_rate * velocity["db" + str(l + 1)]

        return parameters, velocity


# Test function:
parameters = initialize_velocity_test_case()
velocity = initialize_velocity(parameters)
print("velocity[\"dW1\"] = " + str(velocity["dW1"]))
print("velocity[\"db1\"] = " + str(velocity["db1"]))
print("velocity[\"dW2\"] = " + str(velocity["dW2"]))
print("velocity[\"db2\"] = " + str(velocity["db2"]))

parameters, grads, velocity = update_parameters_with_momentum_test_case()
parameters, velocity = update_parameters_with_momentum(
    parameters, grads, velocity,
    beta=0.9, learning_rate=0.01
)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(velocity["dW1"]))
print("v[\"db1\"] = " + str(velocity["db1"]))
print("v[\"dW2\"] = " + str(velocity["dW2"]))
print("v[\"db2\"] = " + str(velocity["db2"]))
