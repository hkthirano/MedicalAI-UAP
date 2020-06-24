import numpy as np
from art.classifiers import KerasClassifier


def set_art(model, norm_str, eps, mean_l2_train, mean_linf_train):
    classifier = KerasClassifier(model=model)
    if norm_str == 'l2':
        norm = 2
        scaled_eps = mean_l2_train / 128.0 * eps
    elif norm_str == 'linf':
        norm = np.inf
        scaled_eps = mean_linf_train / 128.0 * eps
    return classifier, norm, scaled_eps


def get_acc(y, preds):
    acc = np.sum(y == preds) / len(y)
    return acc


def get_fooling_rate(preds, preds_adv):
    fooling_rate = np.sum(preds != preds_adv) / len(preds)
    return fooling_rate


def get_targeted_success_rate(preds_adv, target):
    target_success_rate = np.sum(preds_adv == target) / len(preds_adv)
    return target_success_rate
