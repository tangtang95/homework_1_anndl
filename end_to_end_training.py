from src.data_management.data_reader import *
from src.training.training_with_callbacks import *
from src.model.models import INCRES
import tensorflow as tf

if __name__ == '__main__':
    tf.random.set_seed(get_seed())

    root_path = os.getcwd()
    num_classes = 20

    # Reading the data
    bs=32
    img_h = 256
    img_w = 256
    train_dataset, valid_dataset, train_gen, valid_gen = read_training_data(root_path, bs=bs, img_h=img_h,
                                                                            img_w=img_w, to_rescale=True)

    # Model Creation
    model = INCRES()
    model = model.get_model(img_w=img_w, img_h=img_h)

    # Model training
    exp_dir = os.path.join(root_path, "report")
    callbacks = set_callbacks(exp_dir, model, model.name)


    use_class_weights = True
    if use_class_weights:
        class_w = {0: 1.,
                   1: 1.,
                   2: 500,
                   3: 1.,
                   4: 1.,
                   5: 1.,
                   6: 1.,
                   7: 1.,
                   8: 1.,
                   9: 1.,
                   10 : 500.,
                   11: 1.,
                   12: 1.,
                   13: 1.,
                   14: 4,
                   15: 1.,
                   16: 1.,
                   17: 1.,
                   18: 1.,
                   19: 500.
                   }
    else:
        class_w = None

    model.fit(x=train_dataset,
              epochs=70,
              steps_per_epoch=len(train_gen),
              validation_data=valid_dataset,
              validation_steps=len(valid_gen),
              callbacks=callbacks,
              class_weight=class_w)
