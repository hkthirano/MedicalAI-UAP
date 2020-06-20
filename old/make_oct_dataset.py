import argparse

import numpy as np
import pandas as pd

from utils import make_data

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default='data/oct/train_pre.csv')
parser.add_argument('--test_files', default='data/oct/test_pre.csv')
parser.add_argument(
    '--img_dir', default='../dataset/CellData/OCT')
args = parser.parse_args()

df_train = pd.read_csv(args.train_files, header=None)
df_test = pd.read_csv(args.test_files, header=None)

mapping = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}

X_train, y_train = make_data(
    df_files=df_train,
    img_dir=args.img_dir,
    mapping=mapping,
    dataset='oct')
X_test, y_test = make_data(
    df_files=df_test,
    img_dir=args.img_dir,
    mapping=mapping,
    dataset='oct')

np.save('data/oct/X_train.npy', X_train)
np.save('data/oct/X_test.npy', X_test)
np.save('data/oct/y_train.npy', y_train)
np.save('data/oct/y_test.npy', y_test)
