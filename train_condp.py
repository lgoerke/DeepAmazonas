import os
from argparse import Namespace

import keras
import pickle
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


def train_condp(args):
    model = args.model
    id = args.id
    thres_opt = args.thres_opt
    img_size = args.img_size

    p_train = np.zeros((40479, 17))
    p_test = np.zeros((61191, 17))
    one_third = 13493

    batch_size = 103

    splitter = Validation_splitter('input/train.h5', 1.0 / 3.0)
    print(splitter.num_folds)
    print(splitter.fold_size)

    print('Loading model ', model)
    classifier = keras.models.load_model(os.path.join('models', model))

    for j in range(3):
        splitter.next_fold()

        reader_train = HDF_line_reader('input/train.h5', load_rgb=False, img_size=img_size)
        tg = d.train_generator(reader=reader_train, splitter=splitter, batch_size=batch_size)

        reader_test = HDF_line_reader('input/test.h5', load_rgb=False, img_size=img_size)
        test_g = d.test_generator(reader=reader_test, batch_size=batch_size)

        # print(classifier.model.getShape())
        p_train[one_third * j:one_third * j + one_third, :] = classifier.predict_generator(tg,
                                                                                           one_third // batch_size)
        p_test[one_third * j:one_third * j + one_third, :] = classifier.predict_generator(test_g,
                                                                                          one_third // batch_size)

    all_labels = reader_train.labels
    test_files = reader_test.filenames

    condp2 = pickle.load(open('input/condp2.pkl', 'rb'))
    condp3 = pickle.load(open('input/condp3.pkl', 'rb'))

    cp2flat = np.flatten(condp2)
    cp3flat = np.flatten(condp3)

    predictions_train = np.zeros((40479, 5219))
    predictions_test = np.zeros((61191, 5219))

    for i, line in tqdm(enumerate(p_train)):
        predictions_train[i] = np.concatenate([line, cp2flat, cp3flat])

    for i, line in tqdm(enumerate(p_test)):
        predictions_test[i] = np.concatenate([line, cp2flat, cp3flat])

    model = Sequential()
    model.add(Dense(1024, input_dim=predictions_train.shape[1]))
    model.add(Dense(17, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # define the checkpoint
    filepath = 'ensemble/condp_{}.hdf5'.format(id)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, verbose=1)
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit(predictions_train, all_labels, nb_epoch=100, batch_size=32, callbacks=callbacks_list)

    predictions = model.predict(predictions_test)
    print(predictions.shape)

    result = pd.DataFrame(predictions, columns=data.labels)

    preds = []
    for i in tqdm.tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres_opt, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    df = pd.DataFrame(np.zeros((61191, 2)), columns=['image_name', 'tags'])
    df['image_name'] = test_files
    df['tags'] = preds

    df.to_csv('ensemble/submission_ensemble_{}.csv'.format(id), index=False)


if __name__ == '__main__':
    args = Namespace(model='simple_net_0.68', id='first', thres_opt=0.4, img_size=64)
    train_condp(args)
