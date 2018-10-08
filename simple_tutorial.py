import tensorflow as tf


n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

learing_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(
        X, n_hidden1, name='hidden1', activation=tf.nn.relu
    )
    hidden2 = tf.layers.dense(
        hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu
    )
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

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
