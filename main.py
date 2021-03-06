import numpy as np
import pickle
import sys
import os
import pandas as pd
from tqdm import tqdm
import pdb

import utils
import data_hdf5 as data
from data_hdf5 import Validation_splitter
from data_hdf5 import HDF_line_reader
from sklearn.metrics import fbeta_score

from Classifiers.simple_net import SimpleNet
from Classifiers.densenet import DenseNet

def gen(reader,idx):
    while(True):
        for i in range(0,len(idx),100):
            im, _, _ = reader.read_line_hdf(idx[i:i+100])
            for img in im:
                yield(img[None,...])
        if i < len(idx):
            im, _, _ = reader.read_line_hdf(idx[i:len(idx)])
            for img in im:
                yield(img[None,...])

def main(args):
    size = 64
    #size = 224
    batch_size = 128
    #batch_size = 32
    nb_epoch = 1
    optimizer = 'adam'
    val_split = 0.2
    N_CLASSES = 17
    N_SAMPLES = 40479
    N_TEST = 61191
    test_batch_size = 39

    img_rows, img_cols = size, size # Resolution of inputs
    channel = 3


    labels = data.labels#list(data.LABELS.keys())
    cross_val = True

    #classifier = DenseNet(img_rows, img_cols, batch_size=batch_size, nb_epoch=nb_epoch, color_type=channel, num_classes=N_CLASSES)
    classifier = SimpleNet((size,size,3), n_classes=N_CLASSES, nb_epoch = nb_epoch, batch_size=batch_size, optimizer=optimizer)

    splitter = Validation_splitter('input/train.h5', val_split)
    reader = HDF_line_reader('input/train.h5', load_rgb = False, img_size=size)
    test_reader = HDF_line_reader('input/test.h5', load_rgb = False, img_size=size)

    result = np.zeros((N_TEST,N_CLASSES))
    while(splitter.next_fold() and cross_val):

        tg = data.train_generator(reader, splitter, batch_size)
        vg = data.val_generator(reader, splitter, batch_size)

        print('start training: ')
        classifier.fit(tg, vg, ((1-val_split) * N_SAMPLES, val_split * N_SAMPLES))
        
        print('validating')
        idx = list(splitter.val_idx)
        idx.sort()
                    
        vg = gen(reader, idx)
        #print(next(vg))
        p_valid = classifier.predict(vg, len(splitter.val_idx))
        p_valid = p_valid
        val_labels = reader.labels[idx]

        loss = fbeta_score(val_labels, np.array(p_valid) > 0.2, beta=2, average='samples')
        print('validation loss: {}'.format(loss))

        print('save model:')
        classifier.model.save(os.path.join('models', 'dense_net_{:2.2f}'.format(loss)))

        thres_opt = utils.optimise_f2_thresholds(val_labels, p_valid) 
        
        test_gen = data.test_generator(test_reader, test_batch_size)
        p_test = classifier.predict(test_gen, N_TEST // test_batch_size)
        result += p_test
        
        cross_val = False 

    #result /= splitter.num_folds
    result = pd.DataFrame(result, columns = labels)    
    
    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres_opt, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    df = pd.DataFrame(np.zeros((N_TEST,2)), columns=['image_name','tags'])
    df['image_name'] = test_reader.filenames
    df['tags'] = preds

    id = 1
    df.to_csv('submission{}.csv'.format(id), index=False)


if __name__ == '__main__':
    main([])
