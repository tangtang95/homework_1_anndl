from src.training.training_with_callbacks import *
import tensorflow as tf

from src.utils.utils_functions import get_seed

if __name__ == '__main__':
    tf.random.set_seed(get_seed())
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

    img_w = 256
    img_h = 256

    batch_size = 32
    num_classes = 20

    # ImageDataGenerator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_data_gen = ImageDataGenerator(rotation_range=20,
                                        width_shift_range=1. / 20,
                                        height_shift_range=1. / 20,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        rescale=1. / 255,
                                        cval=0)

    valid_data_gen = ImageDataGenerator(fill_mode='constant',
                                        rescale=1. / 255,
                                        cval=0)

    data_dir = os.path.join(cwd, "data", "Classification_Dataset")

    # Generator creation
    training_dir = os.path.join(data_dir, "training")
    valid_dir = os.path.join(data_dir, "validation")

    train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                                   target_size=(img_w, img_h),
                                                   classes=classes,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=get_seed())

    valid_gen = valid_data_gen.flow_from_directory(directory=valid_dir,
                                                   target_size=(img_w, img_h),
                                                   classes=classes,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   seed=get_seed())

    # Dataset creation
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(tf.float32, tf.float32),
                                                   output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
    train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, output_types=(tf.float32, tf.float32),
                                                   output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
    valid_dataset.repeat()

    # Model Creation
    model_name = "INCRES"
    model = tf.keras.models.Sequential()

    inception_resnet = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                               input_shape=(img_h, img_w, 3))
    inception_resnet.trainable = False

    model.add(inception_resnet)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5, seed=get_seed()))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    model.build(input_shape=(batch_size, img_h, img_w, 3))
    model.summary()

    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Model training
    exp_dir = os.path.join(cwd, "report")
    callbacks = set_callbacks(exp_dir, model, model_name)

    model.fit(x=train_dataset,
              epochs=20,
              steps_per_epoch=len(train_gen),
              validation_data=valid_dataset,
              validation_steps=len(valid_gen),
              callbacks=callbacks)


