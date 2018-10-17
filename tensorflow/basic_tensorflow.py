import numpy as np
import tensorflow as tf


"""
Writing and running programs in Tensorflow has the following steps:
    1. Create Tensors (variables) that are not yet executed/evaluated.
    2. Write operations between those Tensors.
    3. Initialize your Tensors.
    4. Create a Session.
    5. Run the Session. This will run the operations you'd written above.

In the following case, we created a variable for the loss, we simply defined
the loss as a function of other quantities, but did not evaluate its value. To
evaluate it, we had to run init=tf.global_variables_initializer(). That
initialized the loss variable, and in the last line, we were finally able to
evaluate the value of loss and print its value.
"""

y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')
loss = tf.Variable((y - y_hat) ** 2, name='loss')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  # or init.run()
    print(sess.run(loss))


"""
A place holder is an object whose value you can specify later. To specify
values for a placeholder, you can pass in values by using a "feed dictionary"
(fee_dict variable). Below, we created a placeholder for x. This allows us to
pass in a number later when we run the session.

Here is what's happening: when you specify the operations needed for a
computation, you are telling Tensorflow how to construct a computation graph.
The computation graph can have some placeholders whose values you will specify
later. Finally, when you run the session, you are telling Tensorflow to execute
the computation graph.
"""

x = tf.placeholder(tf.int64, name='x')
with tf.Session() as sess:
    print(sess.run(2 * x, feed_dict={x: 3}))

"""
Compute a linear equation Y = WX + b, where W and X are random matrices and b
is a random vector. W is of shape (4, 3), X is (3, 1), and b is (4, 1).
"""


def linear_function():
    """
    Implement a linear function:
        Initialize W to be a random tensor of shape (4, 3)
        Initialize X to be a random tensor of shape (3, 1)
        Initialize b to be a random tensor of shape (4, 1)
    """
    X = tf.constant(np.random.randn(3, 1), name='x')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)

    with tf.Session() as sess:
        result = sess.run(Y)
        print(result)


linear_function()


"""
Compute the sigmoid function of an input, although Tensorflow package provides
a variety of commonly used neural network functions like tf.sigmoid and
tf.softmax.
"""


def sigmoid(z):
    """
    Computes the sigmoid of z

    Argument:
        z : input value, scalar or vector

    Returns:
        results : the sigmoid of z
    """
    x = tf.placeholder(dtype=tf.float32, name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})

    return result


result = sigmoid(0)
print(result)


"""
Compute the cost for logistic regression. You can use a built-in functino to
compute the cost of your neural network. So instead of needing to write code
to compute this as a function of y_hat and y for i = 1...m, you can do it one
line of code in tensorflow.
"""


def compute_cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy

    Arguments:
        vector containing z, output of the last linear unit.
        labels: vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" are respectively called "logits"
    and "labels" in the Tensorflow documentation. So logits will feed into z,
    and labels into y.

    Returns:
        cost : runs the session of the cost
    """
    z = tf.placeholder(dtype=tf.float32, name="logits")
    y = tf.placeholder(dtype=tf.float32, name="labels")

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as sess:
        cost = sess.run(cost, feed_dict={z: logits, y: labels})

    return cost


logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
cost = compute_cost(logits, np.array([0, 0, 1, 1]))
print(cost)
