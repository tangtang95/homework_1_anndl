import os
import tensorflow as tf
from datetime import datetime

if __name__ == '__main__':
    SEED = 482910
    tf.random.set_seed(SEED)

    cwd = os.getcwd()

    classes = [
        'owl',  # 1
        'galaxy',  # 2
        'lightning',  # 3
        'wine-bottle',  # ...
        't-shirt',
        'waterfall',
        'sword',
        'school-bus',
        'calculator',
        'sheet-music',
        'airplanes',
        'lightbulb',
        'skyscraper',
        'mountain-bike',
        'fireworks',
        'computer-monitor',
        'bear',
        'grand-piano',
        'kangaroo',
        'laptop'
    ]

    img_w = 400
    img_h = 400

    batch_size = 32
    num_classes = 20

    # ImageDataGenerator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_data_gen = ImageDataGenerator(rotation_range=20,
                                        width_shift_range=1./20,
                                        height_shift_range=1./20,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0,
                                        rescale=1. / 255)

    valid_data_gen = ImageDataGenerator(rescale=1. / 255,
                                        fill_mode='constant',
                                        cval=0)



    data_dir = os.path.join(cwd, "..", "data", "split")

    # Generator creation
    training_dir = os.path.join(data_dir, "training")
    valid_dir = os.path.join(data_dir, "validation")

    train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                                   target_size=(img_w, img_h),
                                                   classes=classes,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=SEED)

    valid_gen = valid_data_gen.flow_from_directory(directory=valid_dir,
                                                   target_size=(img_w, img_h),
                                                   classes=classes,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   seed=SEED)

    # Dataset creation
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(tf.float32, tf.float32),
                                                   output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
    train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, output_types=(tf.float32, tf.float32),
                                                   output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
    valid_dataset.repeat()

    ########## Model creation ##########
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(img_w, img_h, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    for i in range(8 - 1):
        model.add(tf.keras.layers.Conv2D(filters=32 * (i + 1), kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.05, seed=SEED))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.05, seed=SEED))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    model.build(input_shape=(batch_size, img_w, img_h, 3))
    model.summary()

    # Model compile
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

    ########## Callbacks ##########
    exps_dir = os.path.join(cwd, 'classification_experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    model_name = 'CNN end-to-end training'

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

    ########## Fitting ##########
    model.fit(x=train_dataset,
              epochs=70,
              steps_per_epoch=len(train_gen),
              validation_data=valid_dataset,
              validation_steps=len(valid_gen),
              callbacks=callbacks)

