import itertools

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

from .config import label2nb_dict

matplotlib.use('Agg')


def make_confusion_matrix(y_row, y_col, save_file_name, dataset='melanoma', ylabel='Ground truth', xlabel='Pred clean', title=None):
    from matplotlib import pyplot as plt

    class_label = list(label2nb_dict[dataset].keys())

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


def make_adv_img(clean_img, noise, adv_img, save_file_name):
    # clean
    im_clean = (clean_img * 128.0) + 128.0
    im_clean = np.squeeze(np.clip(im_clean, 0, 255).astype(np.uint8))
    # noise
    im_noise = (noise - noise.min()) / \
        (noise.max() - noise.min()) * 128.0
    im_noise = np.squeeze(im_noise.astype(np.uint8))
    # adv
    im_adv = (adv_img * 128.0) + 128.0
    im_adv = np.squeeze(np.clip(im_adv, 0, 255).astype(np.uint8))
    # all
    img_all = np.concatenate((im_clean, im_noise, im_adv), axis=1)
    img_all = Image.fromarray(np.uint8(img_all))
    img_all.save(save_file_name)
