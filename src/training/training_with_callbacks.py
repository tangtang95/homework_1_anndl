import os
from datetime import datetime
import tensorflow as tf

def retrain_with_callbacks(model_dir, train_dataset, train_gen, valid_dataset, valid_gen, model,
                           n_epochs, last_epoch_num):
    """
    Re-fit the model by setting callbacks and run the fit
    """
    # Set callbacks for retraining
    exp_dir = os.path.join(model_dir, 'retrain', str(last_epoch_num))

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    callbacks = []

    # Model checkpoint
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'),
                                                       save_weights_only=True)  # False to save the training directly
    callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    tb_dir = os.path.join(exp_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                 profile_batch=0,
                                                 histogram_freq=1)  # if 1 shows weights histograms
    callbacks.append(tb_callback)

    # Continue the fit of the model
    model.fit(x=train_dataset,
              epochs=n_epochs,
              steps_per_epoch=len(train_gen),
              validation_data=valid_dataset,
              validation_steps=len(valid_gen),
              callbacks=callbacks)



def get_callbacks(directory, model_name):
    """
    Return the general callbacks to be applied during the fitting of the model

    :param directory: root directory in which to save the experiments (checkpoints, logs, ...)
    :param model_name: name of the model for differentiating the experiments
    :return: callbacks to be used during the fit of the model
    """
    exps_dir = os.path.join(directory, 'classification_experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks = []

    # Model checkpoint
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'),
                                                       save_weights_only=True)  # False to save the training directly
    callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    tb_dir = os.path.join(exp_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                 profile_batch=0,
                                                 histogram_freq=1)  # if 1 shows weights histograms
    callbacks.append(tb_callback)
    return callbacks
