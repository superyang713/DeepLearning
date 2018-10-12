from network import NeuralNetworkClassifier
from propagation import linear_activation
from testCases_v3 import linear_activation_forward_test_case


layer_dims = [5, 4, 3]
nn = NeuralNetworkClassifier(layer_dims)
nn._init_params()
parameters = nn.params

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

A_prev, W, b = linear_activation_forward_test_case()

A, cache = linear_activation(
    A_prev, W, b, activation="sigmoid"
)
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation(
    A_prev, W, b, activation="relu"
)
print("With ReLU: A = " + str(A))
print(cache)
