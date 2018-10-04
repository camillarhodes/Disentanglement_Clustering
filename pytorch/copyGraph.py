from __future__ import print_function
import os
import h5py
import numpy as np
import argparse
import pickle as cPickle
import scipy.io as sio
from config import get_data_dir, get_output_dir

# Note that just like in RCC & RCC-DR, the graph is built on original data.
# Once the features are extracted from the pretrained SDAE,
# they are merged along with the mkNN graph data into a single file using this module.
parser = argparse.ArgumentParser(description='This module is used to merge graph and extracted features into single file')
parser.add_argument('--data', dest='db', type=str, default='mnist', help='name of the dataset')
parser.add_argument('--graph', dest='g', help='path to the graph file', default=None, type=str)
parser.add_argument('--features', dest='feat', help='path to the feature file', default=None, type=str)
parser.add_argument('--out', dest='out', help='path to the output file', default=None, type=str)
parser.add_argument('--h5', dest='h5', action='store_true', help='to store as h5py file')

if __name__ == '__main__':

    args = parser.parse_args()
    datadir = get_data_dir(args.db)
    outputdir = get_output_dir(args.db)

    featurefile = os.path.join(datadir, args.feat)
    graphfile = os.path.join(datadir, args.g)
    outputfile = os.path.join(datadir, args.out)
    if os.path.isfile(featurefile) and os.path.isfile(graphfile):

        if args.h5:
            data0 = h5py.File(featurefile, 'r')
            data1 = h5py.File(graphfile, 'r')
            data2 = h5py.File(outputfile+'.h5','w')
        else:
            fo = open(featurefile,'rb')
            data0 = cPickle.load(fo)
            data1 = sio.loadmat(graphfile)
            fo.close()

        a,b = np.where(data0['data'][:].astype(np.float32).reshape((len(data0['labels'][:]),-1)) -
                       data1['X'][:].astype(np.float32).reshape((len(data1['gtlabels'].T),-1)))
        assert not a.size

        if args.h5:
            data2.create_dataset('gtlabels', data=data0['labels'][:])
            data2.create_dataset('X', data=data0['data'][:].astype(np.float32))
            data2.create_dataset('Z', data=data0['Z'][:].astype(np.float32))
            data2.create_dataset('w', data=data1['w'][:].astype(np.float32))
            data0.close()
            data1.close()
            data2.close()
        else:
            sio.savemat(outputfile+'.mat', {'gtlabels':data0['labels'][:], 'X':data0['data'][:].astype(np.float32),
                                            'Z':data0['Z'][:].astype(np.float32),
                                            'w':data1['w'][:].astype(np.float32)})
    else:
        print('one or both the files not found')
        raise
