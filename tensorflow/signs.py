import tensorflow as tf
import h5py

import matplotlib.pyplot as plt

from tf_utils import load_dataset, convert_to_one_hot
from tf_model import model


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
print(X_train_orig.shape)
print(Y_train_orig.shape)

index = 5
plt.imshow(X_train_orig[index])
plt.show()

# Preprocessing
X_train_flattern = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flattern = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flattern / 255.
X_test = X_test_flattern / 255.

Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

parameters = model(X_train, Y_train, X_test, Y_test)
