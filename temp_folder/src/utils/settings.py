import tensorflow as tf

#SEED = 1234
SEED = 262453 # FROM GAPBN ON

def set_seed():
    # Set the seed for random operations.
    # This let our experiments to be reproducible.
    tf.random.set_seed(SEED)


def get_seed():
    return SEED


def set_gpu():
    # Set GPU memory growth
    # Allows to only as much GPU memory as needed
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
