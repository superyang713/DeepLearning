"""
A simple optimization method in machine learning is gradient descent. When you
take gradient steps with respect to all m samples on each step, it is also
called Batch Gradient Descent.
"""

X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost.
    cost = compute_cost(a, Y)
    # Backward propagation.
    grads = backward_propagation(a, caches, parameters)
    # Update parameters.
    parameters = update_parameters(parameters, grads)


"""
Stochastic Gradient Descent, which is equivalent to mini-batch gradient descent
where each mini-batch has just 1 example. The update rule will not change. What
changes is that you would be computing gradients on just one training sample
at a time, rather than the whole training set. The code examples below
illustrate the difference between stochastic gradient descent and batch
gradient descent.
"""

X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost = compute_cost(a, Y[:,j])
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads)
