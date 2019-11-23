import os
import tensorflow as tf
from datetime import datetime
from src.utils.settings import *
from src.data_management.data_reader import *
from src.training.training_with_callbacks import *
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
    img_h = 410
    img_w = 410
    train_dataset, valid_dataset, train_gen, valid_gen = read_training_data(root_path, bs=bs, img_h=img_h, img_w=img_w)

    model = GAPBN()
    model = model.get_model(bs=bs, img_h=img_h, img_w=img_w)

    model.load_weights(os.path.join(cwd, '..', 'report', 'classification_experiments', 'GAPBN_Nov20_18-25-02',
                                    'retrain', '21','ckpts','cp_35.ckpt'))  # use this if you want to restore saved training


    # Model training
    model_name = "GAPBN"
    exp_dir = os.path.join(root_path, 'report', 'classification_experiments', 'GAPBN_Nov20_18-25-02')
    retrain_with_callbacks(exp_dir, train_dataset, train_gen, valid_dataset, valid_gen, model, model_name, 35, 56)

