"""
Batch and stochastic gradient descent are two extremes for the parameters
update. In practice, you'll get faster results if you do not use neither the
whole training set, nor only one training example, to preform each update.

Mini-batch  gradient descent uses an intermediate number of examples for each
step. With mini-batch gradient descent, you loop over the mini-batches instead
of looping over individual training examples.

There are two steps to build mini-batchs from the training set (X, Y):
    1. Shuffle:
        Create a shuffled version of the training set (X, Y). Each column of X
        and Y represents a training example. Note that the random shuffling is
        done synchronously between X and Y. Such that after the shuffling the
        ith column of X is the example corresponding to the ith label in Y.
        The shuffling step ensures that examples will be split randomly into
        different mini-batches.

    2. Partition:
        Partition shuffled (X, Y) into mini-batches of size mini_batch_size.
        Note that the number of training examples is not always divisible by
        mini_batch_size. The last mini batch might be smaller, but you don't
        need to worry about this.

    3. Handle the end case.
"""


import math
import numpy as np

from testCases import random_mini_batches_test_case


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (n_features, n_samples)
    Y -- true "label" vector of shape (n_outputs, n_samples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)
    n_samples = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    shuffled_indices = np.random.permutation(n_samples)
    shuffled_X = X[:, shuffled_indices]
    shuffled_Y = Y[:, shuffled_indices]

    # Step 2: Partition (shuffled_X, shuffled_Y), excluding the end case.
    n_complete_minibatch = math.floor(n_samples / mini_batch_size)
    for i in range(n_complete_minibatch):
        mini_batch_X = shuffled_X[:, mini_batch_size*i:mini_batch_size*(i+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*i:mini_batch_size*(i+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Step 3: Handle the end case (last mini batch < mini_batch_size)
    if n_samples % mini_batch_size != 0:
        mini_batch_X = shuffled_X[
            :, n_complete_minibatch*mini_batch_size:n_samples]
        mini_batch_Y = shuffled_Y[
            :, n_complete_minibatch*mini_batch_size:n_samples]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# Test function:
X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
