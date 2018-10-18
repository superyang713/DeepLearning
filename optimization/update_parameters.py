"""
Implement the gradient descent update rule. The gradient descent rule is, for
l = 1, ..., L:
    W[l] = W[l] - α * dW[l]
    b[l] = b[l] - α * db[l]

where L is the number of layers (excluding the input layer) and α is the
learning rate. All parameters should be stored in the parameters dictionary.
Note that the iterator l starts at 0 in the for loop while the first parameters
are W[l] and b[l]. Need to shift l to l+1 when coding.
"""


from testCases import update_parameters_with_gd_test_case


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
        parameters['W' + str(l)] = Wl
        parameters['b' + str(l)] = bl
    grads -- python dict containing your gradients to update each parameters:
        grads['dW' + str(l)] = dWl
        grads['db' + str(l)] = dbl
    learning_rate -- scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - \
            learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - \
            learning_rate * grads["db" + str(l + 1)]

    return parameters


# Test function:
parameters, grads, learning_rate = update_parameters_with_gd_test_case()

parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
