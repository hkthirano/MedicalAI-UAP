import os

import cv2
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.data_utils import Sequence
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

FILE_NAME = 0
LABEL = 1

mapping_chestx = {'NORMAL': 0, 'PNEUMONIA': 1}
mapping_oct = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
mapping_melanoma = {'MEL': 0, 'NV': 1, 'BCC': 2,
                    'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
mapping_dict = {'chestx': mapping_chestx,
                'oct': mapping_oct, 'melanoma': mapping_melanoma}


def make_data(df_files, img_dir, dataset='chestx'):
    mapping = mapping_dict[dataset]
    X, y = [], []
    for idx, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
        img_path = os.path.join(img_dir, row[FILE_NAME])
        if dataset == 'chestx':
            img = img_to_array(load_img(img_path, grayscale=True,
                                        color_mode='gray', target_size=(299, 299)))
        elif dataset == 'melanoma':
            img = img_to_array(load_img(img_path, grayscale=False,
                                        color_mode='rgb', target_size=(299, 299)))
        elif dataset == 'oct':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (299, 299))
            img = np.reshape(img, (299, 299, 1))
        X.append(img)
        y.append(mapping[row[LABEL]])
    X = np.asarray(X, dtype='float32')
    y = np.asarray(y)
    y = y.reshape(len(y), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    y = onehot_encoder.fit_transform(y)
    y = np.asarray(y, dtype='float32')
    return X, y


def load_data(dataset='chestx', normalize=True, norm=False):
    X_train = np.load('data/{}/X_train.npy'.format(dataset))
    X_test = np.load('data/{}/X_test.npy'.format(dataset))
    y_train = np.load('data/{}/y_train.npy'.format(dataset))
    y_test = np.load('data/{}/y_test.npy'.format(dataset))

    if normalize:
        X_train = (X_train - 128.0) / 128.0
        X_test = (X_test - 128.0) / 128.0
    return X_train, X_test, y_train, y_test


class BalancedDataGenerator(Sequence):
    """
    ImageDataGenerator + RandomOversampling
    https://medium.com/analytics-vidhya/how-to-apply-data-augmentation-to-deal-with-unbalanced-datasets-in-20-lines-of-code-ada8521320c9
    """

    def __init__(self, x, y, datagen, batch_size=32):
        self.datagen = datagen
        self.batch_size = batch_size
        self._shape = x.shape
        datagen.fit(x)
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(
            x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)

    def __len__(self):
        return self._shape[0] // self.batch_size

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()
