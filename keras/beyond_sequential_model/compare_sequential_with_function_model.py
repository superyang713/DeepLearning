import numpy as np

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input


seq_model = models.Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

# The above model is equivalent to the following functional model
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = models.Model(input_tensor, output_tensor)
model.summary()

# Training is the same for Sequential and Functional model
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy'
)
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))
model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)
print(score)
