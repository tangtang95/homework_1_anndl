from src.data_management.data_reader import *
from src.submission.submission_helper import *
from src.model.models import *
from sklearn.metrics import classification_report
import numpy as np

if __name__ == '__main__':
    # Reading the data
    SEED = 262453

    cwd = os.getcwd()
    root_path = os.path.join(cwd, "..")
    bs=32
    img_h = 410
    img_w = 410
    train_dataset, valid_dataset, train_gen, valid_gen = read_training_data(root_path, SEED, bs=bs, img_h=img_h,
                                                                            img_w=img_w)

    # Model init.
    model = GAPBN()
    model = model.get_model(SEED, bs=bs, img_w=img_w, img_h=img_h)
    model.load_weights(os.path.join(cwd, '..', 'report', 'classification_experiments', 'GAPBN_Nov20_18-25-02',
                                    'retrain', '21','ckpts','cp_35.ckpt'))  # use this if you want to restore saved training


    # Calculating predictions for the validation set
    y_pred = model.predict(x=valid_dataset, steps=len(valid_gen), verbose=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Getting the true classes

    iterator = iter(valid_dataset)

    y_valid = np.zeros((320, 20))
    i = 0
    for _ in range(len(valid_gen)):
        print(i)
        augmented_img, target = next(iterator)
        for j in range(0, bs):
            y_valid[i] = target[j]
            i+=1
    print(i)

    y_valid = y_valid[0:316]
    y_valid = np.argmax(y_valid, axis=1)

    print(classification_report(y_valid, y_pred))