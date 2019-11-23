import os
import random
import numpy as np
import tensorflow as tf

_SEED_ = 69420
_classes_ = [
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


def seed_management():
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = get_seed()

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value

    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value

    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    #session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #K.set_session(sess)


def get_seed():
    return _SEED_


def get_class_list():
    return _classes_
