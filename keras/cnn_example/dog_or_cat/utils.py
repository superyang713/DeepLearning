import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_train_validation_test(
    original_dataset_dir,
    train_cats_dir,
    train_dogs_dir,
    validation_cats_dir,
    validation_dogs_dir,
    test_cats_dir,
    test_dogs_dir,
):
    filenames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for filename in filenames:
        src = os.path.join(original_dataset_dir, filename)
        dest = os.path.join(train_cats_dir, filename)
        shutil.copyfile(src, dest)

    filenames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for filename in filenames:
        src = os.path.join(original_dataset_dir, filename)
        dest = os.path.join(validation_cats_dir, filename)
        shutil.copyfile(src, dest)

    filenames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for filename in filenames:
        src = os.path.join(original_dataset_dir, filename)
        dest = os.path.join(test_cats_dir, filename)
        shutil.copyfile(src, dest)

    filenames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for filename in filenames:
        src = os.path.join(original_dataset_dir, filename)
        dest = os.path.join(train_dogs_dir, filename)
        shutil.copyfile(src, dest)

    filenames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for filename in filenames:
        src = os.path.join(original_dataset_dir, filename)
        dest = os.path.join(validation_dogs_dir, filename)
        shutil.copyfile(src, dest)

    filenames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for filename in filenames:
        src = os.path.join(original_dataset_dir, filename)
        dest = os.path.join(test_dogs_dir, filename)
        shutil.copyfile(src, dest)

    print('total training cat images:', len(os.listdir(train_cats_dir)))
    print('total training dog images:', len(os.listdir(train_dogs_dir)))
    print('total validation cat images:', len(os.listdir(validation_cats_dir)))
    print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
    print('total test cat images:', len(os.listdir(test_cats_dir)))
    print('total test dog images:', len(os.listdir(test_dogs_dir)))


def extract_features(conv_base, data_generator, n_samples):
    """
    Extract features from the images passed by the generator through the pre-
    trained convolutional base model.

    parameters:
    ----------
    conv_base:
        pre-trained convolutional base.
    data_generator:
        generator that passes the mini-batches to the conv-base.
    n_samples: int
        number of samples in the data.
    """
    features = np.zeros(shape=(n_samples, 4, 4, 512))   # shape of VGG16 output
    labels = np.zeros(shape=(n_samples,))
    i = 0
    for inputs_batch, labels_batch in data_generator:
        batch_size = len(labels_batch)
        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch

        i += 1
        if i * batch_size >= n_samples:
            break
    return features, labels


def plot_loss_and_accuracy(history):
    # Plot training and validation loss and accuracy
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    epochs = range(1, len(loss_values) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.plot(epochs, loss_values, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss_values, 'r', label='Validation loss')
    ax1.set_title('Training and Validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, acc_values, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_acc_values, 'r', label='Validation accuracy')
    ax2.set_title('Training and Validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    plt.show()


def display_augmented_image(datagen, image_path, targe_size=(150, 150)):
    """
    Display what images look like after a single image is augmented.

    Parameters:
    -----------
    datagen: instance of ImageDataGenerator class.
    image_path: string of an image file path.
    targe_size: tuple of an image size.
    """
    img = image.load_img(image_path, target_size=targe_size)
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    fig, axes = plt.subplots(2, 2)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        axes[i // 2, i % 2].imshow(image.array_to_img(batch[0]))
        axes[i // 2, i % 2].axis('off')
        i += 1
        if i % 4 == 0:
            break
    plt.show()


def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    directory = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'image'
    )
    if not os.path.isdir(directory):
        os.mkdir(directory)
    fig_path = os.path.join(directory, fig_id + '.' + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)


if __name__ == "__main__":
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data'
    )
    train_dir = os.path.join(base_dir, 'train')
    train_cats_dir = os.path.join(train_dir, 'cats')
    fnames = [os.path.join(train_cats_dir, fname)
              for fname in os.listdir(train_cats_dir)]
    display_augmented_image(datagen, fnames[3])
