import itertools

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

matplotlib.use('Agg')


def plotConfMat(y_row, y_col, save_file_name, dataset='melanoma', ylabel='Ground truth', xlabel='Pred clean', title=None):
    from matplotlib import pyplot as plt

    if 'chestx' in dataset:
        class_label = ['NORMAL', 'PNEUMONIA']
    elif 'oct' in dataset:
        class_label = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    elif 'melanoma' in dataset:
        class_label = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    cm = pd.DataFrame(confusion_matrix(y_row, y_col))
    cm = cm.values
    cm = cm.astype('float32')
    for i in range(len(class_label)):
        cm[i, :] /= cm[i, :].sum()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_label))
    plt.xticks(tick_marks, class_label, rotation=45)
    plt.yticks(tick_marks, class_label)
    plt.title(title)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(save_file_name)
    plt.close()
