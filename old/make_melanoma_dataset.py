import argparse

import numpy as np
import pandas as pd

from utils import make_data

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default='data/melanoma/train_pre.csv')
parser.add_argument('--test_files', default='data/melanoma/test_pre.csv')
parser.add_argument(
    '--img_dir', default='../dataset/ISIC2018_Task3_Training_Input')
parser.add_argument(
    '--img_gt_csv', default='../dataset/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
args = parser.parse_args()

mapping = {'MEL': 0, 'NV': 1, 'BCC': 2,
           'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}

df_train = pd.read_csv(args.train_files, header=None)
df_test = pd.read_csv(args.test_files, header=None)

X_train, y_train = make_data(
    df_files=df_train,
    img_dir=args.img_dir,
    mapping=mapping,
    dataset='melanoma')
X_test, y_test = make_data(
    df_files=df_test,
    img_dir=args.img_dir,
    mapping=mapping,
    dataset='melanoma')

np.save('data/melanoma/X_train.npy', X_train)
np.save('data/melanoma/X_test.npy', X_test)
np.save('data/melanoma/y_train.npy', y_train)
np.save('data/melanoma/y_test.npy', y_test)
