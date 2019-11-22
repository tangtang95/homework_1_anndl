import os
import tensorflow as tf
from datetime import datetime
from src.utils.settings import *
from src.data_management.data_reader import *
from src.training.training_with_callbacks import *

if __name__ == '__main__':
    SEED = 262453
    tf.random.set_seed(SEED)
    cwd = os.getcwd()
    root_path = os.path.join(cwd, "..")
    num_classes = 20

    # Reading the data
    bs=32
    img_h = 410
    img_w = 410
    train_dataset, valid_dataset, train_gen, valid_gen = read_training_data(root_path, bs=bs, img_w=img_w, img_h=img_h)

    # Model Creation
    model_name = "GAPBN"
    model = tf.keras.models.Sequential()
    total_depth = 6

    # Feature extraction
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(img_w, img_h, 3),
                                     use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    for i in range(1, total_depth - 1):
        model.add(tf.keras.layers.Conv2D(filters=32 * (2**i), kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        if i <= 4:
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last'))

    # First FC layer
    model.add(tf.keras.layers.Dense(units=256, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5, seed=SEED))

    # Second FC layer
    model.add(tf.keras.layers.Dense(units=128, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5, seed=SEED))

    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    model.build(input_shape=(bs, img_w, img_h, 3))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

    model.summary()

    # Model training
    exp_dir = os.path.join(root_path, "report")
    train_with_callbacks(exp_dir, train_dataset, train_gen, valid_dataset, valid_gen, model, model_name)

