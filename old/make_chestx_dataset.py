import argparse

import numpy as np
import pandas as pd

from utils import make_data

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default='data/chestx/train_pre.csv')
parser.add_argument('--test_files', default='data/chestx/test_pre.csv')
parser.add_argument(
    '--img_dir', default='../dataset/CellData/chest_xray')
args = parser.parse_args()

df_train = pd.read_csv(args.train_files, header=None)
df_test = pd.read_csv(args.test_files, header=None)

mapping = {'NORMAL': 0, 'PNEUMONIA': 1}

X_train, y_train = make_data(
    df_files=df_train,
    img_dir=args.img_dir,
    mapping=mapping,
    dataset='chestx')
X_test, y_test = make_data(
    df_files=df_test,
    img_dir=args.img_dir,
    mapping=mapping,
    dataset='chestx')

np.save('data/chestx/X_train.npy', X_train)
np.save('data/chestx/X_test.npy', X_test)
np.save('data/chestx/y_train.npy', y_train)
np.save('data/chestx/y_test.npy', y_test)
