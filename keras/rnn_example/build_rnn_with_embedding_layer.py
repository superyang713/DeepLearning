from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing

from tensorflow.keras import models
from tensorflow.keras import layers

from utils import plot_loss_and_accuracy


n_features = 10000
n_timesteps = 20

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=n_features)
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=n_timesteps)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=n_timesteps)

# Now X_train shape is (n_samples, n_timesteps), each integer in the matrix
# representing a word.
print(X_train.shape)

# Build the model
model = models.Sequential()

# Embedding layer output shape (n_samples, n_timesteps, 8)
model.add(layers.Embedding(10000, 8, input_length=n_timesteps))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_split=0.2
)

# Visualize
plot_loss_and_accuracy(history)
