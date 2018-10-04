import os
import os.path as osp
from config import cfg, get_data_dir

import random
import argparse
import numpy as np
import scipy.io as sio
import h5py

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_mnist(root, training):
    if training:
        data = 'train-images-idx3-ubyte'
        label = 'train-labels-idx1-ubyte'
        N = 60000
    else:
        data = 't10k-images-idx3-ubyte'
        label = 't10k-labels-idx1-ubyte'
        N = 10000
    with open(osp.join(root,data), 'rb') as fin:
        fin.seek(16, os.SEEK_SET)
        X = np.fromfile(fin, dtype=np.uint8).reshape((N,28*28))
    with open(osp.join(root,label), 'rb') as fin:
        fin.seek(8, os.SEEK_SET)
        Y = np.fromfile(fin, dtype=np.uint8)
    return X, Y

def make_mnist_data(path, isconv=False):
    X, Y = load_mnist(path, True)
    X = X.astype(np.float64)
    X2, Y2 = load_mnist(path, False)
    X2 = X2.astype(np.float64)
    X3 = np.concatenate((X,X2), axis=0)

    minmaxscale = MinMaxScaler().fit(X3)

    X = minmaxscale.transform(X)
    if isconv:
        X = X.reshape((-1,1,28,28))

    sio.savemat(osp.join(path,'traindata.mat'), {'X':X, 'Y':Y})

    X2 = minmaxscale.transform(X2)
    if isconv:
        X2 = X2.reshape((-1, 1, 28, 28))

    sio.savemat(osp.join(path,'testdata.mat'), {'X':X2, 'Y':Y2})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', dest='db', type=str, default='mnist', help='name of the dataset')

    args = parser.parse_args()
    np.random.seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)

    datadir = get_data_dir(args.db)
    strpath = osp.join(datadir,'traindata.mat')

    if not os.path.exists(strpath):
        if args.db == 'mnist':
            make_mnist_data(datadir)
        else:
            print("db not supported: '{}'".format(args.db))
            raise

