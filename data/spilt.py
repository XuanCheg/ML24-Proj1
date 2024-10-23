import scipy.io as scio
from sklearn.model_selection import train_test_split
import numpy as np
import os
from os.path import join, exists


def split_data(data_name, test_size=0.3):
    data = scio.loadmat(data_name)
    data_train = {}
    data_test = {}
    data_val = {}
    keys = [d for d in data.keys() if '__' not in d]
    print(keys)
    for key in keys:
        d = np.array(data[key])
        d_train, d_valtest = train_test_split(d, test_size=test_size, random_state=2024)
        d_test, d_val = train_test_split(d_valtest, test_size=0.5, random_state=2024)
        data_train[key] = d_train
        data_test[key] = d_test
        data_val[key] = d_val
    data_name = data_name.split('.')[0]
    if not exists(data_name):
        os.mkdir(data_name)
    np.savez(join(data_name, 'train.npz'), **data_train)
    np.savez(join(data_name, 'test.npz'), **data_test)
    np.savez(join(data_name, 'val.npz'), **data_val)


if __name__ == '__main__':
    for i in ['ADNI_90_120_fMRI.mat', 'ADNI.mat', 'FTD_90_200_fMRI.mat', 'OCD_90_200_fMRI.mat', 'PPMI.mat']:
        print(f'Spillting {i}...')
        split_data(i)
