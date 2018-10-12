from network import NeuralNetworkClassifier
from propagation import linear_activation
from test import (
    linear_activation_forward_test_case, L_model_forward_test_case_2hidden,
    compute_cost_test_case
)


layer_dims = [5, 4, 3]
nn = NeuralNetworkClassifier(layer_dims)
nn._init_params()
parameters = nn.params

print('-' * 50)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

A_prev, W, b = linear_activation_forward_test_case()

A, cache = linear_activation(
    A_prev, W, b, activation="sigmoid"
)
print('-' * 50)
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation(
    A_prev, W, b, activation="relu"
)
print('-' * 50)
print("With ReLU: A = " + str(A))
print(cache)

X, parameters = L_model_forward_test_case_2hidden()
nn.params = parameters
y_hat = nn._forward_propagate(X)
print('-' * 50)
print("y_hat = " + str(y_hat))
print("Length of caches list = " + str(len(nn.caches)))

print('-' * 50)
y, y_hat = compute_cost_test_case()
print("cost = " + str(nn._compute_cost(y_hat, y)))
