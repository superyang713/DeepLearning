"""
Use pre-trained model for feature extraction also with data augmentation.
Extend the conv_base model and run it end to end on the inputs.

Note:
    This technique is so expensive that you should only attempt it if you have
    access to GPU. If without GPU, the other feature extraction method is the
    way to go.
"""


import os

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16

from utils import plot_loss_and_accuracy


# File locations
base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data'
)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)

conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3),
)


# Build the model with dropout
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc']
)


# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
)
model.save('catas_and_dogs_small_4.h5')

# Visualize the performance
plot_loss_and_accuracy(history)
