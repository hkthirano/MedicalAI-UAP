import argparse

import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from utils.config import set_gpu
from utils.data import BalancedDataGenerator, load_data
from utils.model import load_model
from utils.plot import make_confusion_matrix
from utils.utils import get_acc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chestx')
parser.add_argument('--model', type=str, default='inceptionv3')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

set_gpu(args.gpu)

X_train, X_test, y_train, y_test = load_data(
    dataset=args.dataset, normalize=True)

model = load_model(
    dataset=args.dataset,
    nb_class=y_train.shape[1],
    model_type=args.model,
    mode='train'
)


def step_decay(epoch):
    lr = 1e-3
    if epoch > 45:
        lr = 1e-5
    elif epoch > 40:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr


epoch = 50
batch_size = 32
lr_decay = LearningRateScheduler(step_decay)
save_model = 'data/{}/model/{}.h5'.format(args.dataset, args.model)
cb1 = ModelCheckpoint(save_model, monitor='val_acc', verbose=1,
                      save_best_only=True, save_weights_only=True)
datagen = ImageDataGenerator(rotation_range=5,
                             width_shift_range=0.05,
                             height_shift_range=0.05)

if args.dataset == 'melanoma':
    balanced_gen = BalancedDataGenerator(
        X_train, y_train, datagen, batch_size=batch_size)
    model.fit_generator(
        balanced_gen,
        steps_per_epoch=balanced_gen.steps_per_epoch,
        epochs=epoch,
        validation_data=(X_test, y_test),
        callbacks=[lr_decay, cb1],
        verbose=1)
else:
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=epoch,
                        validation_data=(X_test, y_test),
                        callbacks=[lr_decay, cb1],
                        verbose=1)

# make confusion matrix
preds_train = np.argmax(model.predict(X_train), axis=1)
preds_test = np.argmax(model.predict(X_test), axis=1)

acc_train = get_acc(y=np.argmax(y_train, axis=1), preds=preds_train)
acc_test = get_acc(y=np.argmax(y_test, axis=1), preds=preds_test)

save_f_train = 'data/{}/conf_mat/train_{}.png'.format(
    args.dataset, args.model)
save_f_test = 'data/{}/conf_mat/test_{}.png'.format(
    args.dataset, args.model)

make_confusion_matrix(
    y_row=np.argmax(y_train, axis=1),
    y_col=preds_train,
    save_file_name=save_f_train,
    dataset=args.dataset,
    title='acc_train : {:.3f}'.format(acc_train)
)
make_confusion_matrix(
    y_row=np.argmax(y_test, axis=1),
    y_col=preds_test,
    save_file_name=save_f_test,
    dataset=args.dataset,
    title='acc_test : {:.3f}'.format(acc_test)
)
