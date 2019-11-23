from src.data_management.data_reader import *
from src.training.training_with_callbacks import *
from src.model.models import *

if __name__ == '__main__':
    # Initial settings: https://github.com/keras-team/keras/issues/1920
    tf.random.set_seed(get_seed())
    root_path = os.getcwd()
    num_classes = 20

    # Reading the data
    bs=32
    img_h = 256
    img_w = 256
    train_dataset, valid_dataset, train_gen, valid_gen = read_training_data(root_path, bs=bs, img_h=img_h,
                                                                            img_w=img_w)

    model = INCRES()
    model = model.get_model(batch_size=bs, img_w=img_w, img_h=img_h)

    model.load_weights(os.path.join(root_path, 'report', 'classification_experiments', 'INCRES_Nov23_14-08-54',
                                    'ckpts','cp_07.ckpt'))  # use this if you want to restore saved training


    # Model training
    model_name = "INCRES"
    exp_dir = os.path.join(root_path, 'report', 'classification_experiments', 'INCRES_Nov23_14-08-54')
    retrain_with_callbacks(exp_dir, train_dataset, train_gen, valid_dataset, valid_gen, model, model_name, 35, 8)

