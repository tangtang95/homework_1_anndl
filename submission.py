import os
import tensorflow as tf
from datetime import datetime
from src.utils.settings import *
from src.data_management.data_reader import *
from src.training.training_with_callbacks import *
from src.submission.submission_helper import *
from src.model.models import *

if __name__ == '__main__':
    # Initial settings: https://github.com/keras-team/keras/issues/1920
    SEED = get_seed()
    tf.random.set_seed(SEED)
    cwd = os.getcwd()
    root_path = os.path.join(cwd, "..")
    num_classes = 20

    # Reading the data
    bs=32
    img_h = 256
    img_w = 256
    test_dataset, test_gen = read_test_data(root_path, bs=bs, img_h=img_h, img_w=img_w)

    # RSN1 INIT
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
    model.load_weights(os.path.join(cwd, '..', 'report', 'classification_experiments', 'RSN1_Nov21_22-59-47',
                                    'ckpts', 'cp_08.ckpt'))

    # INC1 INIT
    inc1 = INC1()
    inc1 = inc1.get_model()
    inc1.load_weights(os.path.join(cwd, '..', 'report', 'classification_experiments', 'INC_Nov22_17-28-11',
                                    'ckpts', 'cp_07.ckpt'))

    # INC2 INIT
    inc2 = INC2()
    inc2 = inc2.get_model()
    inc2.load_weights(os.path.join(cwd, '..', 'report', 'classification_experiments', 'INC2_Nov22_18-56-41',
                                   'ckpts', 'cp_07.ckpt'))


    # ENSAMBLE INIT
    model_list = []
    model_list.append(model)
    model_list.append(inc1)
    model_list.append(inc2)
    ensamble_avg = EnsembleBestPrediction(model_list, 500)

    test_dataset_list = []
    test_dataset_list.append(test_dataset)
    test_dataset_list.append(test_dataset)
    test_dataset_list.append(test_dataset)

    # Model test
    predict_submissions(ensamble_avg, test_dataset_list, test_gen)