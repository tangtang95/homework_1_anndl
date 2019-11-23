from src.data_management.data_reader import *
from src.submission.submission_helper import *
from src.model.models import *

if __name__ == '__main__':
    tf.random.set_seed(get_seed())
    cwd = os.getcwd()
    num_classes = 20

    # Reading the data
    bs=32
    img_h = 256
    img_w = 256
    test_dataset, test_gen = read_test_data(cwd, bs=bs, img_h=img_h, img_w=img_w)

    model = INCRES()
    model = model.get_model()
    model.load_weights(os.path.join(cwd, 'report', 'classification_experiments', 'INCRES_Nov23_14-08-54',
                                    'retrain', '8', 'ckpts', 'cp_01.ckpt'))

    # Model test
    predict_submissions(model, test_dataset, test_gen)