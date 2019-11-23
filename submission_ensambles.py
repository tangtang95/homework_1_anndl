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

    # RSN1 INIT
    rsn = RSN1()
    rsn = rsn.get_model()
    rsn.load_weights(os.path.join(cwd, 'report', 'classification_experiments', 'RSN1_Nov21_22-59-47',
                                    'ckpts', 'cp_08.ckpt'))

    # INC1 INIT
    inc1 = INC1()
    inc1 = inc1.get_model()
    inc1.load_weights(os.path.join(cwd, 'report', 'classification_experiments', 'INC_Nov22_17-28-11',
                                    'ckpts', 'cp_07.ckpt'))

    # INC2 INIT
    inc2 = INC2()
    inc2 = inc2.get_model()
    inc2.load_weights(os.path.join(cwd, 'report', 'classification_experiments', 'INC2_Nov22_18-56-41',
                                   'ckpts', 'cp_07.ckpt'))


    # ENSAMBLE INIT
    model_list = []
    model_list.append(rsn)
    model_list.append(inc1)
    model_list.append(inc2)
    ensemble_avg = EnsembleBestPrediction(model_list, 500)

    test_dataset_list = []
    test_dataset_list.append(test_dataset)
    test_dataset_list.append(test_dataset)
    test_dataset_list.append(test_dataset)

    # Model test
    predict_submissions(ensemble_avg, test_dataset_list, test_gen)