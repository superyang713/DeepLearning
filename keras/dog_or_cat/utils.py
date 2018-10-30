import os
import shutil


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
