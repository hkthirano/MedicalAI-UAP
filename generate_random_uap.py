import argparse

import numpy as np
from art.utils import random_sphere

from utils.config import label2nb_dict, set_gpu
from utils.data import load_data
from utils.model import load_model
from utils.plot import make_adv_img, make_confusion_matrix
from utils.utils import get_fooling_rate, get_targeted_success_rate, set_art

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chestx')
parser.add_argument('--model', type=str, default='inceptionv3')
parser.add_argument('--norm', type=str, default='l2')
parser.add_argument('--eps', type=float, default=0.04)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

set_gpu(args.gpu)

X_train, X_test, y_train, y_test, mean_l2_train, mean_inf_train = load_data(
    dataset=args.dataset, normalize=True, norm=True)

model = load_model(
    dataset=args.dataset,
    nb_class=y_train.shape[1],
    model_type=args.model,
    mode='inference'
)

# # Generate adversarial examples

classifier, norm, eps = set_art(
    model, args.norm, args.eps, mean_l2_train, mean_inf_train)

h, w, c = X_train.shape[1], X_train.shape[2], X_train.shape[3]
noise = random_sphere(nb_points=1,
                      nb_dims=(h * w * c),
                      radius=eps,
                      norm=norm)
noise = noise.reshape(h, w, c).astype('float32')

base_f = 'random_{}_{}_eps{:.3f}'.format(
    args.model, args.norm, args.eps)
save_f_noise = 'result/{}/noise/{}'.format(args.dataset, base_f)
np.save(save_f_noise, noise)

# # Evaluate the ART classifier on adversarial examples

preds_train = np.argmax(classifier.predict(X_train), axis=1)
preds_test = np.argmax(classifier.predict(X_test), axis=1)

X_train_adv = X_train + noise
X_test_adv = X_test + noise

preds_train_adv = np.argmax(classifier.predict(X_train_adv), axis=1)
preds_test_adv = np.argmax(classifier.predict(X_test_adv), axis=1)

rf_train = get_fooling_rate(preds=preds_train, preds_adv=preds_train_adv)
rf_test = get_fooling_rate(preds=preds_test, preds_adv=preds_test_adv)

rs_train_list, rs_test_list = [], []
label_list = label2nb_dict[args.dataset].keys()
for target in label_list:
    rs_train_list.append(get_targeted_success_rate(
        preds_train_adv, label2nb_dict[args.dataset][target]))
    rs_test_list.append(get_targeted_success_rate(
        preds_test_adv, label2nb_dict[args.dataset][target]))

save_f_train = 'result/{}/conf_mat/train_{}.png'.format(
    args.dataset, base_f)
save_f_test = 'result/{}/conf_mat/test_{}.png'.format(
    args.dataset, base_f)

title_train = 'Rf:{:.3f}\nRs '.format(rf_train)
title_test = 'Rf:{:.3f}\nRs '.format(rf_test)

for i, target in enumerate(label_list):
    title_train = title_train + \
        '{}:{:.3f}  '.format(target, rs_train_list[i])
    title_test = title_test + \
        '{}:{:.3f}  '.format(target, rs_train_list[i])

make_confusion_matrix(
    y_row=preds_train,
    y_col=preds_train_adv,
    save_file_name=save_f_train,
    dataset=args.dataset,
    ylabel='Pred clean',
    xlabel='Pred adv',
    title=title_train
)
make_confusion_matrix(
    y_row=preds_test,
    y_col=preds_test_adv,
    save_file_name=save_f_test,
    dataset=args.dataset,
    ylabel='Pred clean',
    xlabel='Pred adv',
    title=title_test
)

# # Show the adversarial examples

save_f_img = 'result/{}/imshow/{}.png'.format(args.dataset, base_f)
make_adv_img(
    clean_img=X_test[0],
    noise=noise,
    adv_img=X_test_adv[0],
    save_file_name=save_f_img
)
