import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return (train_set_x_orig,
            train_set_y_orig,
            test_set_x_orig,
            test_set_y_orig,
            classes)


def plot_image(X, y, index):
    """
    Visualize an example image in the dataset.
    """
    plt.imshow(X[index])
    plt.axis('off')
    plt.show()


def describe_data(X_train_orig, y_train, X_test_orig, y_test):
    n_samples_train = X_train_orig.shape[0]
    n_samples_test = X_test_orig.shape[0]
    print('-' * 50)
    print("Number of training samples: {}".format(n_samples_train))
    print("Number of testing samples: {}".format(n_samples_test))
    print("X_train_orig shape: {}".format(X_train_orig.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test_orig shape: {}".format(X_test_orig.shape))
    print("y_test shape: {}".format(X_test_orig.shape))
    print('-' * 50)


def reshape_X(X):
    """
    The original shape of X is (n_samples, height, width, color), and for \
        neural network training and testing, it should be reshaped \
        (n_features, n_samples).
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1).T


def sigmoid(z):
    """
    Compute the sigmoid of z.

    Parameters:
    ----------
    z -- A scalar or numpy array of any size.

    Return:
    ------
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(n_features):
    """
    This function creates a vector of zeros of shape (1, n_features) for w \
        and initializes b to 0.

    Parameters:
    ----------
    n_features : size of the w vector we want (or number of parameters in \
        this case)

    Returns:
    -------
    w : initialized vector of shape (1, n_features)
    b : initialized scalar (corresponds to the bias)
    """
    w = np.zeros((1, n_features))
    b = 0

    assert(w.shape == (1, n_features))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


def forward_propagate(w, b, X, y):
    """
    Implement the cost function for the propagation.

    Parameters:
    ----------
    w : weights vector, a numpy array of size (1, n_features)
    b : bias, a scalar
    X : data, a numpy array of size (n_features, n_samples)
    y : label vecotr, a numpy array of size (1, n_samples)

    Return:
    -------
    A : activation, a numpy array of size (1, n_samples)
    cost : negative log-likelihood cost for logistic regression
        For logistic regression, the gradient does not depend on cost value.
        but it is still good to keep it.
    """
    n_samples = X.shape[1]
    A = sigmoid(np.dot(w, X) + b)
    cost = -1 / n_samples * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

    return A, cost


def backward_progagate(w, b, X, y):
    """
    Find the gradient of the cost function for the backward propagation.

    Parameters:
    ----------
    w : weights vector, a numpy array of size (1, n_features)
    b : bias, a scalar
    X : data, a numpy array of size (n_features, n_samples)
    y : label vecotr, a numpy array of size (1, n_samples)

    Return:
    -------
    grads : dict
        A dictionary containing the gradient of the loss with respect to w \
            and b.
    """

    n_samples = X.shape[1]
    n_features = X.shape[0]
    A, cost = forward_propagate(w, b, X, y)
    dw = 1 / n_samples * np.dot(A - y, X.T)
    db = 1 / n_samples * np.sum(A - y)

    assert(dw.shape == (1, n_features))
    assert(db.dtype == float)

    grads = {'dw': dw, 'db': db}

    return grads


def optimize(w, b, X, y, num_iterations, learning_rate, print_cost=False):
    """
    Optimizes w and b by running a gradient decent algorithm.

    Parameters:
    ----------
    w : weights, a numpy array of size (1, n_features)
    b : bias, a scalar
    X : data, a numpy array of size (n_features, n_samples)
    y : label vecotr, a numpy array of size (1, n_samples)
    num_iterations : number of iterations of the optimization loop
    learning_rate : learning rate of the gradient descent update rule
    print_cost : True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the optimized w and b
    """
    costs = []
    for i in range(num_iterations):
        _, cost = forward_propagate(w, b, X, y)
        grads = backward_progagate(w, b, X, y)
        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration {}: {}".format(i, cost))

    params = {"w": w, "b": b}
    return params


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression \
        parameters (w, b)

    Parameters:
    ----------
    w : weights, a numpy array of size (1, n_features)
    b : bias, a scalar
    X : data, a numpy array of size (n_features, n_samples)

    Returns:
    -------
    y_predict : a numpy array (vector) containing all predictions.
    '''
    n_samples = X.shape[1]
    A = sigmoid(np.dot(w, X) + b)

    y_predict = np.rint(A)
    y_predict = y_predict.reshape(1, -1)
    assert(y_predict.shape == (1, n_samples))

    return y_predict


def get_accuracy(y_predict, y):
    n_samples = y.shape[1]
    accuracy = np.sum(y_predict == y) / n_samples
    return accuracy
