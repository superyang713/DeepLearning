from utils import (
    load_dataset, plot_image, describe_data, reshape_X, forward_propagate,
    backward_progagate, optimize, predict, initialize_with_zeros,
    get_accuracy
)


# Load data.
X_train_orig, y_train, X_test_orig, y_test, classes = load_dataset()

# plot_image(X_train_orig, y_train, 1)

# Check data shape.
describe_data(X_train_orig, y_train, X_test_orig, y_test)

# flattern feature array.
X_train_flattern = reshape_X(X_train_orig)
X_test_flattern = reshape_X(X_test_orig)

# Standardize feature array.
X_train = X_train_flattern / 255.
X_test = X_test_flattern / 255.


# ------------------------test -----------------------------
#w, b, X, Y = np.array([1.,2.]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])


#cost = forward_propagate(w, b, X, Y)
#print(cost)

# grads = backward_progagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))

n_features = X_train.shape[0]
w, b = initialize_with_zeros(n_features)
params = optimize(w, b, X_train, y_train, num_iterations=2000,
                  learning_rate = 0.009, print_cost = True)
y_predict_train = predict(params['w'], params['b'], X_train)
y_predict_test = predict(params['w'], params['b'], X_test)
train_accuracy = get_accuracy(y_predict_train, y_train)
test_accuracy = get_accuracy(y_predict_test, y_test)
print("train accuracy: {}".format(train_accuracy))
print("test accuracy: {}".format(test_accuracy))

# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
#
#
# w = np.array([0.1124579,0.23106775])
# b = -0.3
# X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
# print ("predictions = " + str(predict(w, b, X)))


