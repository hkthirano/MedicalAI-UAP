import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def set_gpu(gpu):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=gpu,
            allow_growth=True))
    set_session(tf.Session(config=config))


def get_acc(y, pred):
    acc = np.sum(y == pred) / len(y)
    return acc
