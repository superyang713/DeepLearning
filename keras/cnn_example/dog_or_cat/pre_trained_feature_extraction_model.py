"""
Fast feature extraction without data augmentation.
Running the pre-trained convolutional base over the dataset, recording its
output to a numpy array on disk, and then using this data as imput to a
standalone, densely connected classifier. This solution is fast and cheap to
run, because it only requires running the convolutional base once for every
input image, and the convolutional base is by far the most expensive part of
the pipeline. But for the same reason, this technique does not allow you to use
data augmentation.
"""


import os

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16

from utils import plot_loss_and_accuracy
from utils import extract_features


# Organize files into different folders
base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data'
)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)

conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3),
)

train_features, train_labels = extract_features(
    conv_base, train_generator, 2000
)
validation_features, validation_labels = extract_features(
    conv_base, validation_generator, 1000
)
test_features, test_labels = extract_features(
    conv_base, test_generator, 1000
)

train_features = train_features.reshape(2000, 4 * 4 * 512)
validation_features = validation_features.reshape(1000, 4 * 4 * 512)
test_features = test_features.reshape(1000, 4 * 4 * 512)

# Build the model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc'],
)

# Train the model
history = model.fit(
    train_features,
    train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features, validation_labels),
)
model.save('catas_and_dogs_small_3.h5')

# Visualize the performance
plot_loss_and_accuracy(history)
