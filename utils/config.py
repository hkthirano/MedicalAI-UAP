import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

label2nb_dict = {
    'chestx':
        {'NORMAL': 0, 'PNEUMONIA': 1},
    'oct':
        {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3},
    'melanoma':
        {'MEL': 0, 'NV': 1, 'BCC': 2,
         'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
}


def set_gpu(gpu):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=gpu,
            allow_growth=True))
    set_session(tf.Session(config=config))
