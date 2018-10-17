import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

from tf_utils import random_mini_batches


def model(
    X_train, Y_train,
    X_test, Y_test,
    learning_rate=0.0001,
    num_epochs=1500,
    minibatch_size=32,
    print_cost=True
):
    """
    Implements a three-layer tensorflow neural network:
        LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape
        (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape
        (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape
        (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape
        (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model.
        They can then be used to predict.
    """

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    # Construction Phase
    X, Y = create_placeholder(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate
    ).minimize(cost)

    init = tf.global_variables_initializer()

    # Execution Phase
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(
                X_train, Y_train, minibatch_size, seed
            )

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run(
                    [optimizer, cost],
                    feed_dict={X: minibatch_X, Y: minibatch_Y}
                )

                epoch_cost += minibatch_cost / num_minibatches
            if print_cost and epoch % 100 == 0:
                print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy: ", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy: ", accuracy.eval({X: X_test, Y: Y_test}))

        plt.show()

        return parameters


def create_placeholder(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector
        (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes
        (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input
    Y -- placeholder for the input labels

    Tips:
    - You will use None because it let's us be flexible on the number of
    examples you will for the placeholders. In fact, the number of examples
    during test/train is different.
    """

    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None])

    return X, Y


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow.
    The shapes are:
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)

    xavier_initializer = tf.contrib.layers.xavier_initializer(seed=1)
    zero_initializer = tf.zeros_initializer()

    W1 = tf.get_variable("W1", [25, 12288], initializer=xavier_initializer)
    b1 = tf.get_variable("b1", [25, 1], initializer=zero_initializer)
    W2 = tf.get_variable("W2", [12, 25], initializer=xavier_initializer)
    b2 = tf.get_variable("b2", [12, 1], initializer=zero_initializer)
    W3 = tf.get_variable("W3", [6, 12], initializer=xavier_initializer)
    b3 = tf.get_variable("b3", [6, 1], initializer=zero_initializer)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
        LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters
        "W1", "b1", "W2", "b2", "W3", "b3"

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit),
        of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # Tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(..,..)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )

    return cost
