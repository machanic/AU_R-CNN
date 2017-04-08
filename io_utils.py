import sys
import numpy as np
import cv2
import h5py
import lmdb
import caffe

from caffe.proto import caffe_pb2
from config import *


@DeprecationWarning
def gen_hd5(filename, data_set):
    '''
    Note that data_set is a dict ,whose value is (abs_path,label) tuple
    '''
    setname = filename
    sample_size = len(data_set)
    imgs = numpy.zeros((sample_size, 1,) + IMG_SIZE, dtype=numpy.float32)
    labels = numpy.zeros(sample_size, dtype=numpy.int32)
    h5_filename = '{0}/{1}.h5'.format(BACKEND_DIR,setname)
    with h5py.File(h5_filename, 'w') as h:
        for i, (abs_path, label) in enumerate(data_set.values()):
             #opencv automatically read image as BGR,which is caffe prefer
            img = cv2.imread(abs_path).astype("int")
            img = img.reshape((1, )+img.shape)
            img -= MEAN_VALUE
            imgs[i] = img
            labels[i] = int(label)
            if (i+1) % 1000 == 0:
                print('processed {} images!'.format(i+1))
        h.create_dataset('data', data=imgs)
        h.create_dataset('label', data=labels)

    with open('{0}/{1}_h5.txt'.format(BACKEND_DIR, setname), 'w') as f:
        f.write(h5_filename)


def gen_lmdb(filename, data_set):
    '''
    Note that lmdb use incremental index as key
    '''
    setname = filename
    sample_size = len(data_set)
    db_path = "{0}/{1}.lmdb".format(BACKEND_DIR,setname)
    batch_size = 200

    lmdb_env = lmdb.open(db_path, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin(write=True)
    datum = caffe_pb2.Datum()
    for i, (abs_path, label) in enumerate(data_set.values()):
         #opencv automatically read image as BGR,which is caffe prefer, img' shape is height, width, channel
        img = cv2.resize(cv2.imread(abs_path), IMG_SIZE).astype("int")
        img = np.transpose(img , (2, 0, 1)) # Caffe's shape is N C H W, so must switch
        #img = img.reshape((1, )+ IMG_SIZE)
        #logger.log("img path {0} label {1}".format(abs_path, label))
        datum = caffe.io.array_to_datum(img, label)
        keystr = '{:0>8d}'.format(i)
        lmdb_txn.put(keystr, datum.SerializeToString())
        # write batch
        if (i+1) % batch_size == 0:
            lmdb_txn.commit()
            lmdb_txn = lmdb_env.begin(write=True)
            logger.log("batch {} written".format(i))
            
    lmdb_txn.commit()
    lmdb_env.close()