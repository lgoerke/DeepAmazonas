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


def main(args):
    size = 64
    batch_size = 96
    nb_epoch = 1
    optimizer = 'adadelta'
    val_split = 0.2
    N_CLASSES = 17
    N_SAMPLES = 40479
    N_TEST = 61191
    test_batch_size = 9

    labels = list(data.LABELS.keys())
    cross_val = True

    classifier = SimpleNet((size,size,4), n_classes=N_CLASSES, nb_epoch = nb_epoch, batch_size=batch_size, optimizer=optimizer)

    splitter = Validation_splitter('input/train.h5', val_split)
    #test_data, file_ids = data.get_all_test('input/test-tif-v2', img_size=size, load_rgb=True)
    #val_data, val_labels = data.get_all_val('input/train-tif-v2', reader, splitter, img_size=size, load_rgb=True)

    result = np.zeros((N_TEST,N_CLASSES))
    while(splitter.next_fold() and cross_val):

        reader = HDF_line_reader('input/train.h5', load_rgb = False, img_size=size)
        tg = data.train_generator(reader, splitter, batch_size)
        vg = data.val_generator(reader, splitter, batch_size)

        print('start training: ')
        classifier.fit(tg, vg, ((1-val_split) * N_SAMPLES, val_split * N_SAMPLES))
        
        print('validating')
        #pdb.set_trace()
        p_valid = classifier.evaluate(vg, len(splitter.val_idx))

        print('validation loss: {}'.format(fbeta_score(val_labels, np.array(p_valid) > 0.2, beta=2, average='samples')))

        print('save model:')
        classifier.model.save(os.path.join('models', 'simple_net_{}'.format(loss)))

        thres_opt = utils.optimise_f2_thresholds(val_labels, p_valid) 
        
        test_reader = HDF_line_reader('input/test.h5', load_rgb = False, img_size=size)
        test_gen = data.test_generator(reader, test_batch_size)
        p_test = classifier.predict(test_data, N_TEST // test_batch_size)
        result += p_test
        
        cross_val = False 

    result /= splitter.num_folds
    result = pd.DataFrame(result, columns = labels)    
    
    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres_opt, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    result['tags'] = preds
    result['image_name'] = file_ids

    result.to_csv('submission_keras.csv', index=False)


if __name__ == '__main__':
    main([])
