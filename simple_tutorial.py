import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np


n_inputs = 8 * 8
n_hidden1 = 100
n_hidden2 = 50
n_hidden3 = 20
n_outputs = 10

learing_rate = 0.01

mnist = load_digits()
data = mnist['data']
target = mnist['target']

X_train, X_test, y_train, y_test = (
    data[:1500], data[1500:], target[:1500], target[1500:]
)

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(
        X, n_hidden1, name='hidden1', activation=tf.nn.relu
    )
    hidden2 = tf.layers.dense(
        hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu
    )
    hidden3 = tf.layers.dense(
        hidden2, n_hidden3, name='hidden3', activation=tf.nn.relu
    )
    logits = tf.layers.dense(hidden3, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits
    )
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learing_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execution phase
n_epochs = 40
batch_size = 50
n_samples = X_train.shape[0]
n_batches = int(np.ceil(n_samples / batch_size))
print(n_batches)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            X_batch = X_train[iteration*batch_size:(iteration+1)*batch_size]
            y_batch = y_train[iteration*batch_size:(iteration+1)*batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
