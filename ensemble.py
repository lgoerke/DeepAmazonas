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

    all_data, all_labels = d.get_all_train('input/train-tif-v2', reader, splitter, img_size=256, load_rgb=False)
    all_test, test_files = d.get_all_test('input/test-tif-v2', reader, splitter, img_size=256, load_rgb=False)

    predictions_train = np.zeros((len(modellist), len(all_data), 17))
    predictions_test = np.zeros((len(modellist), len(all_test), 17))

    for i, m in enumerate(modellist):
        classifier = keras.model.load(os.path.join('models', m))
        predictions_train[i, :, :] = classifier.predict(all_data)
        predictions_test[i, :, :] = classifier.predict(all_test)

    return predictions_train, all_labels, predictions_test, test_files


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
        predictions = np.mean(predictions_test, axis=0)
        print(predictions.shape)
    elif mode == 'max':
        predictions = np.max(predictions_test, axis=0)
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

    mlist = args.modellist
    val_split = args.val_split
    mode = args.mode
    id = args.id
    thres_opt = args.thres_opt
    csv_files = args.csv_files

    if mlist:
        predictions_train, all_labels, predictions_test, test_files = create_predictions(mlist, val_split)
    else:
        test_files = d.get_all_test_files('input/test-tif-v2')
        predictions_test = np.zeros((0, 61191, 17))

    if mode == "network":
        assert isinstance(mlist, object, "Network mode needs models")
        train_ensemble(predictions_train, all_labels, id)

    if csv_files:
        if mode == "network":
            print('CSV files are ignored because of network mode')
        else:
            predictions_tmp = np.zeros(
                (predictions_test.shape[0] + len(csv_files), predictions_test.shape[1], predictions_test.shape[2]))
            predictions_csv = np.zeros((len(csv_files), predictions_test.shape[1], predictions_test.shape[2]))
            for i, c in enumerate(csv_files):
                df = pd.read_csv(c)
                for j, row in enumerate(df['tags']):
                    predictions_csv[i, j, :] = d.to_one_hot(row)
        predictions_tmp[:predictions_test.shape[0], :, :] = predictions_test
        predictions_tmp[predictions_test.shape[0]:, :, :] = predictions_csv
        predictions_test = predictions_tmp

    predictions = predict_with_ensemble(predictions_test, mode)

    result = pd.DataFrame(predictions, columns=labels)

    preds = []
    for i in tqdm.tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres_opt, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    df = pd.DataFrame(np.zeros((61191,2)), columns=['image_name','tags'])
    df['image_name'] = test_files
    df['tags'] = preds

    df.to_csv('ensemble/submission_ensemble_{}.csv'.format(id), index=False)


if __name__ == '__main__':
    mlist = []
    csv_files = ['input/submissions/submission_1.csv', 'input/submissions/submission_blend.csv']
    val_split = 0.2
    mode = 'mean'
    id = '2ndTry'
    thres_opt = 0.4
    args = Namespace(val_split=val_split, modellist=mlist, mode=mode, id=id, thres_opt=thres_opt, csv_files=csv_files)

    ensemble(args)
