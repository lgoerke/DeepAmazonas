import os
from argparse import Namespace

import keras
import numpy as np
import pandas as pd
import tqdm
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.models import Sequential

import data as d
from data import CSV_line_reader
from data import Validation_splitter


def create_predictions(modellist, val_split):
    splitter = Validation_splitter('input/train_v2.csv', val_split)
    splitter.next_fold()

    reader = CSV_line_reader('input/train_v2.csv')

    all_data, all_labels = d.get_all_train('input/train_v2.csv', reader, splitter, img_size=256, load_rgb=False)
    all_test = d.get_all_test('input/test_v2.csv', reader, splitter, img_size=256, load_rgb=False)

    predictions_train = np.zeros((len(modellist, len(all_data), 17)))
    predictions_test = np.zeros((len(modellist, len(all_data), 17)))

    for i, m in enumerate(modellist):
        classifier = keras.model.load(os.path.join('models', m))
        predictions_train[i, :, :] = classifier.predict(all_data)
        predictions_test[i, :, :] = classifier.predict(all_test)

    return predictions_train, all_labels, predictions_test


def train_ensemble(predictions_train, all_labels, id):
    model = Sequential()
    model.add(Dense(1024, input_shape=(predictions_train.shape[0] * 17,)))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # define the checkpoint
    filepath = 'ensemble/ensemble_{}.hdf5'.format(id)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, verbose=1)
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit(predictions_train, all_labels, nb_epoch=100, batch_size=32, callbacks=callbacks_list)


def predict_with_ensemble(predictions_test, mode='mean'):
    if mode == 'network':
        model = Sequential()
        model.add(Dense(1024, input_shape=(predictions_test.shape[0] * 17,)))
        model.add(Dense(8, activation='softmax'))

        # define the checkpoint
        filepath = 'ensemble/ensemble_{}.hdf5'.format(id)
        model.load_weights(filepath)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        predictions = model.predict(predictions_test)
        print(predictions.shape)
    elif mode == "mean":
        predictions = np.mean(predictions_test, axis=1)
        print(predictions.shape)
    elif mode == 'max':
        predictions = np.max(predictions_test, axis=1)
        print(predictions.shape)

    return predictions


def ensemble(args):
    labels = ['blow_down',
              'bare_ground',
              'conventional_mine',
              'blooming',
              'cultivation',
              'artisinal_mine',
              'haze',
              'primary',
              'slash_burn',
              'habitation',
              'clear',
              'road',
              'selective_logging',
              'partly_cloudy',
              'agriculture',
              'water',
              'cloudy']

    mlist = args.mlist
    val_split = args.val_split
    mode = args.mode
    id = args.id
    thres_opt = args.thres_opt

    predictions_train, all_labels, predictions_test = create_predictions(mlist, val_split)

    if mode == "network":
        train_ensemble(predictions_train, all_labels, id)

    predictions = predict_with_ensemble(predictions_test, mode)

    result = pd.DataFrame(predictions, columns=labels)

    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres_opt, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    result['tags'] = preds

    result.to_csv('submission_ensemble{}.csv'.format(id), index=False)


if __name__ == '__main__':
    mlist = ['simple_netwhatever', 'yo']
    val_split = 0.2
    mode = 'mean'
    id = 'firstTry'
    thres_opt = 0.5
    args = Namespace(val_split=val_split, modellist=mlist, mode=mode, id=id, thres_opt=thres_opt)

    ensemble(args)
