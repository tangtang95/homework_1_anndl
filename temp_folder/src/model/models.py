import tensorflow as tf
from src.utils.settings import get_seed
import numpy as np

class GAPBN():
    def __init__(self):
        self.model_name = "GAPBN"

    def get_model(self, bs=32, img_w=410, img_h=410, num_classes=20):
        SEED=get_seed()

        model = tf.keras.models.Sequential()
        total_depth = 6

        # Feature extraction
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(img_w, img_h, 3),
                                         use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        for i in range(1, total_depth - 1):
            model.add(tf.keras.layers.Conv2D(filters=32 * (2 ** i), kernel_size=(3, 3), padding='same', use_bias=False))
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

        return model


class GAP2():
    def __init__(self):
        self.model_name = "GAP2"

    def get_model(self, bs = 32, img_w=440, img_h=440, num_classes=20):
        model_name = "GAP2"
        SEED = 262453
        model = tf.keras.models.Sequential()
        total_depth = 6

        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(img_w, img_h, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        for i in range(1, total_depth - 1):
            model.add(
                tf.keras.layers.Conv2D(filters=32 * (2 ** i), kernel_size=(3, 3), padding='same', activation='relu'))
            if i <= 4:
                model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last'))
        model.add(tf.keras.layers.Dense(units=256, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.1, seed=SEED))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.05, seed=SEED))
        model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

        model.build(input_shape=(bs, img_w, img_h, 3))
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

        model.build(input_shape=(bs, img_w, img_h, 3))
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

        return model

class CNN1():
    def __init__(self):
        self.model_nome = "CNN1"

    def get_model(self, num_classes, bs, img_w, img_h, seed):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(img_w, img_h, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        for i in range(8 - 1):
            model.add(
                tf.keras.layers.Conv2D(filters=32 * (i + 1), kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.05, seed=seed))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.05, seed=seed))
        model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

        model.build(input_shape=(bs, img_w, img_h, 3))
        model.summary()

        # Model compile
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

        return model


class FCNN1():

    def __init__(self):
        self.model_name = "FCNN2"

    def get_model(self, num_classes, bs, img_w, img_h):
        model = tf.keras.models.Sequential()
        depth_max_pool = 4
        total_depth = 8

        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), padding='same', input_shape=(img_w, img_h, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        for i in range(1, total_depth - 1):
            model.add(tf.keras.layers.Conv2D(filters=32 * i, kernel_size=(4, 4), padding='same', activation='relu'))
            if i <= depth_max_pool:
                model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

        model.build(input_shape=(bs, img_w, img_h, 3))
        optimizer = tf.keras.optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

        return model

class NetworkAverageEnsemble(object):
    def __init__(self, model_list, num_items_to_predict):
        self.model_list = model_list
        self.num_models = len(model_list)
        self.num_items_to_predict = num_items_to_predict

    def predict(self, x=None, steps=None, verbose=1, num_classes=20):
        predictions = np.zeros((self.num_models, self.num_items_to_predict, num_classes))

        # Computing predictions
        for i, model in enumerate(self.model_list):
            predictions[i] = model.predict(x=x[i], steps=steps, verbose=verbose)

        # Averaging predictions
        y_pred = np.average(predictions, axis=0)

        return y_pred

class EnsembleBestPrediction(object):
    def __init__(self, model_list, num_items_to_predict):
        self.model_list = model_list
        self.num_models = len(model_list)
        self.num_items_to_predict = num_items_to_predict

    def predict(self, x=None, steps=None, verbose=1, num_classes=20):
        predictions = np.zeros((self.num_models, self.num_items_to_predict, num_classes))

        # Computing predictions
        for i, model in enumerate(self.model_list):
            predictions[i] = model.predict(x=x[i], steps=steps, verbose=verbose)

        # Averaging predictions
        y_pred = np.zeros((self.num_items_to_predict, num_classes))
        for i in range(0, self.num_items_to_predict):
            max = 0
            best_model_pred = predictions[0][i]
            for j in range(1, self.num_models):
                temp = np.argmax(predictions[j][i])
                if temp > max:
                    best_model_pred = predictions[j][i]
            y_pred[i] = best_model_pred

        return y_pred

class RCN1(object):

    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self, seed, num_classes=20, bs=32, img_h=256, img_w=256):
        model_name = "RSN1"
        resnet = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_tensor=None,
                                                            input_shape=(img_w, img_h, 3), pooling='avg', classes=1000)
        resnet.trainable = False
        model = tf.keras.Sequential()
        model.add(resnet)
        model.add(tf.keras.layers.Dense(units=256, activation='elu'))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

class INC1(object):

    def get_model(self, seed, img_w=256, img_h=256, num_classes=20):
        bs = 32
        img_h = 256
        img_w = 256

        # Model Creation
        inception = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
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

        return model



class INC2(object):

    def get_model(self, seed, img_w=256, img_h=256, num_classes=20):
        bs = 32
        img_h = 256
        img_w = 256

        # Model Creation
        inception = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                   weights='imagenet',
                                                                   input_tensor=None,
                                                                   input_shape=(img_w, img_h, 3),
                                                                   pooling='avg',
                                                                   classes=1000)

        inception.trainable = False
        model = tf.keras.Sequential()
        model.add(inception)
        model.add(tf.keras.layers.Dense(units=512, activation='elu'))
        model.add(tf.keras.layers.Dropout(rate=0.4))
        model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model


class INCRES():
    def get_model(self, seed, img_w=256, img_h=256, num_classes=20):
        inception_resnet = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                   input_shape=(img_h, img_w, 3))
        inception_resnet.trainable = False


        model = tf.keras.Sequential()
        model.add(inception_resnet)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.5, seed=seed))
        model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
