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
    img_h = 256
    img_w = 256
    train_dataset, valid_dataset, train_gen, valid_gen = read_training_data(root_path, bs=bs, img_w=img_w, img_h=img_h,
                                                                            to_rescale=True)

    # Model Creation
    model_name = "INCRES"
    inception = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                            weights='imagenet',
                                                            input_tensor=None,
                                                            input_shape=(img_w, img_h, 3),
                                                            pooling='avg',
                                                            classes=1000)

    inception.trainable = False
    model = tf.keras.Sequential()
    model.add(inception)
    model.add(tf.keras.layers.Dense(units=256, activation='elu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    #model.summary()



