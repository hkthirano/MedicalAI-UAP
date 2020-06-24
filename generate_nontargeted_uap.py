import argparse

import numpy as np
from art.attacks import UniversalPerturbation

from utils.config import set_gpu
from utils.data import load_data
from utils.model import load_model
from utils.plot import make_adv_img, make_confusion_matrix
from utils.utils import get_fooling_rate, set_art

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chestx')
parser.add_argument('--model', type=str, default='inceptionv3')
parser.add_argument('--norm', type=str, default='l2')
parser.add_argument('--eps', type=float, default=0.04)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

set_gpu(args.gpu)

X_train, X_test, y_train, y_test, mean_l2_train, mean_linf_train = load_data(
    dataset=args.dataset, normalize=True, norm=True)

model = load_model(
    dataset=args.dataset,
    nb_class=y_train.shape[1],
    model_type=args.model,
    mode='inference'
)

# # Generate adversarial examples

classifier, norm, eps = set_art(
    model=model,
    norm_str=args.norm,
    eps=args.eps,
    mean_l2_train=mean_l2_train,
    mean_linf_train=mean_linf_train)

adv_crafter = UniversalPerturbation(
    classifier,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted': False, 'eps': 0.0024},
    max_iter=15,
    eps=eps,
    norm=norm)

_ = adv_crafter.generate(X_train)
noise = adv_crafter.noise[0, :].astype(np.float32)
base_f = 'nontargeted_{}_{}_eps{:.3f}'.format(
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

save_f_train = 'result/{}/conf_mat/train_{}.png'.format(
    args.dataset, base_f)
save_f_test = 'result/{}/conf_mat/test_{}.png'.format(
    args.dataset, base_f)

make_confusion_matrix(
    y_row=preds_train,
    y_col=preds_train_adv,
    save_file_name=save_f_train,
    dataset=args.dataset,
    ylabel='Pred clean',
    xlabel='Pred adv',
    title='fooling_rate_train : {:.3f}'.format(rf_train)
)
make_confusion_matrix(
    y_row=preds_test,
    y_col=preds_test_adv,
    save_file_name=save_f_test,
    dataset=args.dataset,
    ylabel='Pred clean',
    xlabel='Pred adv',
    title='fooling_rate_test : {:.3f}'.format(rf_test)
)

# # Show the adversarial examples

save_f_img = 'result/{}/imshow/{}.png'.format(args.dataset, base_f)
make_adv_img(
    clean_img=X_test[0],
    noise=noise,
    adv_img=X_test_adv[0],
    save_file_name=save_f_img
)
