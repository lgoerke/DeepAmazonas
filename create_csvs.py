import os
from argparse import Namespace

import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.models import Sequential

import data_hdf5 as d
import data as data
from data_hdf5 import HDF_line_reader
from data_hdf5 import Validation_splitter
import tensorflow as tf


def create_predictions(modellist, img_sizes):
    predictions_train = np.zeros((len(modellist) , 40479, 17))
    predictions_test = np.zeros((len(modellist) , 61191, 17))

    for i, m in enumerate(modellist):


        one_third = 13493
        batch_size = 103

        splitter = Validation_splitter('input/train.h5', 1.0 / 3.0)
        print(splitter.num_folds)
        print(splitter.fold_size)

        print('Loading model ', m)
        if i > 0:
            del classifier

        with tf.variable_scope(m):
            classifier = keras.models.load_model(os.path.join('models', m))

        print('Create predictions train')
        for j in tqdm(range(3)):
            splitter.next_fold()

            reader_train = HDF_line_reader('input/train.h5', load_rgb=False, img_size=img_sizes[i])
            tg = d.train_generator(reader=reader_train, splitter=splitter, batch_size=batch_size)

            # print(classifier.model.getShape())
            predictions_train[i, one_third * j:one_third * j + one_third, :] = classifier.predict_generator(tg,
                                                                                                            one_third // batch_size)

        one_third = 20397
        batch_size = 523

        splitter = Validation_splitter('input/test.h5', 1.0 / 3.0)
        print(splitter.num_folds)
        print(splitter.fold_size)

        for j in tqdm(range(3)):
            reader_test = HDF_line_reader('input/test.h5', load_rgb=False, img_size=img_sizes[i])
            test_g = d.test_generator(reader=reader_test, batch_size=batch_size)

            predictions_test[i, one_third * j:one_third * j + one_third, :] = classifier.predict_generator(test_g,
                                                                                                           one_third // batch_size)

    return predictions_train, reader_train.labels, predictions_test, reader_train.filenames, reader_test.filenames



def create_csvs(args):
    mlist = args.modellist
    img_sizes = args.img_sizes

    predictions_train, all_labels, predictions_test, train_files, test_files = create_predictions(mlist, img_sizes)
    pdf_names = pd.DataFrame(test_files, columns=['image_name'])
    for idx, p in enumerate(predictions_train):
        pdf = pd.DataFrame(p, columns=[*data.labels])
        pdf = pd.concat([pdf_names, pdf], axis=1)
        pdf.sort_values('image_name')
        pdf.to_csv('ensemble/predictions_train_{}.csv'.format(mlist[idx]), index=False)
        pdf.drop(pdf.index, inplace=True)

    for idx, p in enumerate(predictions_test):
        pdf = pd.DataFrame(p, columns=[*data.labels])
        pdf = pd.concat([pdf_names, pdf], axis=1)
        pdf.sort_values('image_name')
        pdf.to_csv('ensemble/predictions_test_{}.csv'.format(mlist[idx]), index=False)
        pdf.drop(pdf.index, inplace=True)


if __name__ == '__main__':
    mlist = ['simple_net_048','simple_net_068']
    # List with same size as mlist
    img_sizes = [64,64]
    args = Namespace(modellist=mlist, img_sizes = img_sizes)

    create_csvs(args)