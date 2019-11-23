import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils.utils_functions import *


def read_test_data(root_path, bs, img_h, img_w):
    test_data_gen = ImageDataGenerator(rescale=1. / 255, fill_mode='constant', cval=0)

    data_dir = os.path.join(root_path, "data", "Classification_Dataset")

    test_gen = test_data_gen.flow_from_directory(directory=data_dir,
                                                 classes=['test'],
                                                 target_size=(img_w, img_h),
                                                 shuffle=False,
                                                 batch_size=bs,
                                                 seed=get_seed()
                                                 )

    test_dataset = tf.data.Dataset.from_generator(lambda: test_gen, output_types=(tf.float32, tf.float32),
                                                  output_shapes=([None, img_h, img_w, 3], [None, None]))
    test_dataset.repeat()

    return test_dataset, test_gen


def read_training_data(root_path, bs=32, img_h=256, img_w=256, to_rescale=True):
    # Define some variables
    classes = get_class_list()
    num_classes = len(classes)

    if to_rescale:
        rescale_factor = 1. / 255
    else:
        rescale_factor = None

    train_data_gen = ImageDataGenerator(rotation_range=20,
                                        width_shift_range=1. / 20,
                                        height_shift_range=1. / 20,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0,
                                        rescale=rescale_factor)

    valid_data_gen = ImageDataGenerator(rescale=rescale_factor,
                                        fill_mode='constant',
                                        cval=0)

    data_dir = os.path.join(root_path, "data", "split")

    # Generator creation
    training_dir = os.path.join(data_dir, "training")
    valid_dir = os.path.join(data_dir, "validation")

    train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                                   target_size=(img_w, img_h),
                                                   classes=classes,
                                                   batch_size=bs,
                                                   shuffle=True,
                                                   seed=get_seed())

    valid_gen = valid_data_gen.flow_from_directory(directory=valid_dir,
                                                   target_size=(img_w, img_h),
                                                   classes=classes,
                                                   batch_size=bs,
                                                   shuffle=False,
                                                   seed=get_seed())

    # Dataset creation
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(tf.float32, tf.float32),
                                                   output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, output_types=(tf.float32, tf.float32),
                                                   output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
    train_dataset.repeat()
    valid_dataset.repeat()
    return train_dataset, valid_dataset, train_gen, valid_gen
