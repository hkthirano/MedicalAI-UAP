import argparse

import numpy as np
import pandas as pd

from utils.data import make_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='chestx')
parser.add_argument(
    '--img_dir', default='../dataset/CellData/chest_xray')
args = parser.parse_args()

df_train = pd.read_csv(
    'data/{}/train_pre.csv'.format(args.dataset),
    header=None)
df_test = pd.read_csv(
    'data/{}/test_pre.csv'.format(args.dataset),
    header=None)

X_train, y_train = make_data(
    df_files=df_train,
    img_dir=args.img_dir,
    dataset=args.dataset)
X_test, y_test = make_data(
    df_files=df_test,
    img_dir=args.img_dir,
    dataset=args.dataset)

np.save('data/{}/X_train.npy'.format(args.dataset), X_train)
np.save('data/{}/X_test.npy'.format(args.dataset), X_test)
np.save('data/{}/y_train.npy'.format(args.dataset), y_train)
np.save('data/{}/y_test.npy'.format(args.dataset), y_test)
