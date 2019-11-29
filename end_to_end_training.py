from src.data_management.data_reader import *
from src.training.training_with_callbacks import *
from src.model.models import GAPBN
import tensorflow as tf

if __name__ == '__main__':
    tf.random.set_seed(get_seed())

    root_path = os.getcwd()
    num_classes = 20

    # Reading the data
    bs = 32
    img_h = 256
    img_w = 256
    train_dataset, valid_dataset, train_gen, valid_gen = read_training_data(root_path, batch_size=bs, img_h=img_h,
                                                                            img_w=img_w, to_rescale=True)

    # Model Creation
    model = GAPBN().get_model(img_w=img_w, img_h=img_h)

    # Model training
    exp_dir = os.path.join(root_path, "report")
    callbacks = get_callbacks(exp_dir, model.name)

    model.fit(x=train_dataset,
              epochs=70,
              steps_per_epoch=len(train_gen),
              validation_data=valid_dataset,
              validation_steps=len(valid_gen),
              callbacks=callbacks)
