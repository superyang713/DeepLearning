"""
Create an RNN composed of a layer of five recurrent neurons, using the tanh
activation function. In this case, the RNN runs over only two time steps,
taking input vectors of size 3 at each time step. The following code builds
this RNN, unrolled through two time steps:
"""


import tensorflow as tf
import numpy as np


X0_batch = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 0, 1],
])

X1_batch = np.array([
    [9, 8, 7],
    [0, 0, 0],
    [6, 5, 4],
    [3, 2, 1],
])

n_features = X1_batch.shape[1]
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_features])
X1 = tf.placeholder(tf.float32, [None, n_features])

Wx = tf.Variable(
    tf.random_normal(shape=[n_features, n_neurons], dtype=tf.float32)
)
Wy = tf.Variable(
    tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32)
)
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    feed_dict = {
        X0: X0_batch,
        X1: X1_batch
    }
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict=feed_dict)


"""
The following code in higher tensorflow api creates the exact model as the
previous one.
"""

X0 = tf.placeholder(tf.float32, [None, n_features])
X1 = tf.placeholder(tf.float32, [None, n_features])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(
    basic_cell, [X0, X1], dtype=tf.float32
)

Y0, Y1 = output_seqs

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    feed_dict = {
        X0: X0_batch,
        X1: X1_batch
    }
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict=feed_dict)

print(Y0_val)
