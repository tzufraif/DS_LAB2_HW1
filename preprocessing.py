import pandas as pd
import random
from data_EDA import *
import numpy as np
import os


def preprocess(path, mode='train'):
    """
    Preprocess of the given data files in the given path. Including null/NA handling, transformations, eliminations,
    interpolation, and label inference.
    :param path: A path to data files
    :param mode: If mode is set to 'train' so down sampling will be performed
    :return:
    """
    X = np.array(np.zeros((1, 78)))
    for file_name in os.listdir(path):
        df = pd.read_csv(f"{path}/{file_name}", sep='|')
        df = df.drop(columns=df.iloc[:, 7:34].columns, axis=1)
        df = df.drop(['Unit1', 'Unit2'], axis=1)
        df = df.interpolate(method='linear')
        df = df.fillna(df.mean())
        indices = df[df['SepsisLabel'] == 1].index
        label = 0 if len(indices) == 0 else 1
        df = df.drop(indices[1:])
        describe = df.iloc[:, :-1].describe()  # ignores labels
        row = describe.T.iloc[:, 1:]
        row = row.to_numpy().flatten()
        row = np.append(row, label)
        X = np.vstack([X, row])
    # now df is a vector with mean representation for each column
    print('@@@@@@@@@@@@@@@@@@@@ Finished! @@@@@@@@@@@@@@@@@@@@')
    X = np.delete(X, 0, 0)
    X = pd.DataFrame(X)
    X.iloc[:, :-1] = (X.iloc[:, :-1] - X.iloc[:, :-1].mean()) / X.iloc[:, :-1].std()

    # for col in X:
    #     X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.interpolate(method='linear')
    X = X.fillna(X.mean())
    # df.dropna(axis=1, inplace = True)
    X.drop(57, axis=1, inplace=True)
    if mode == 'train':
        X = down_sampling(X)

    return X.iloc[:, :-1], X.iloc[:, -1]