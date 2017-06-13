import os
from argparse import Namespace

import keras
import numpy as np
import pandas as pd
import tqdm
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.models import Sequential

import data_hdf5 as d
import data as data
from data_hdf5 import HDF_line_reader
from data_hdf5 import Validation_splitter


def create_predictions(modellist, val_split):

    predictions_train = np.zeros((len(modellist), 40479, 17))
    predictions_test = np.zeros((len(modellist), 61191, 17))
    one_third = 13493

    batch_size=103

    for i, m in enumerate(modellist):
        splitter = Validation_splitter('input/train.h5', 1.0/3.0)
        print(splitter.num_folds)
        print(splitter.fold_size)

        print('Loading model ', m)
        classifier = []
        classifier = keras.models.load_model(os.path.join('models', m))

        for j in range(3):
            splitter.next_fold()

            reader_train = HDF_line_reader('input/train.h5', load_rgb=False, img_size=img_sizes[i])
            tg = d.train_generator(reader=reader_train,splitter=splitter,batch_size=batch_size)

            reader_test = HDF_line_reader('input/test.h5', load_rgb=False, img_size=img_sizes[i])
            test_g =d.test_generator(reader=reader_test,batch_size=batch_size)

            #print(classifier.model.getShape())
            predictions_train[i, one_third*j:one_third*j+one_third, :] = classifier.predict_generator(tg,one_third//batch_size)
            predictions_test[i, one_third*j:one_third*j+one_third, :] = classifier.predict_generator(test_g,one_third//batch_size)

    return predictions_train, reader_train.labels, predictions_test, reader_test.filenames


def train_ensemble(predictions_train, all_labels, id):
    predictions_train = np.transpose(predictions_train,(1,0,2))
    predictions_train = np.reshape(predictions_train,(predictions_train.shape[0], predictions_train[1]*predictions_train[2]))

    model = Sequential()
    model.add(Dense(1024, input_dim=(predictions_train.shape[1])))
    model.add(Dense(17, activation='softmax'))
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
        predictions_test = np.transpose(predictions_test, (1, 0, 2))
        predictions_test = np.reshape(predictions_test,
                                       (predictions_test.shape[0], predictions_test[1] * predictions_test[2]))

        model = Sequential()
        model.add(Dense(1024, input_dim=(predictions_test.shape[1])))
        model.add(Dense(17, activation='softmax'))

        # define the checkpoint
        filepath = 'ensemble/ensemble_{}.hdf5'.format(id)
        model.load_weights(filepath)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        predictions = model.predict(predictions_test)
        print(predictions.shape)
    elif mode == "mean":
        predictions = np.mean(predictions_test, axis=0)
        print('== Check mean ==')
        print(predictions.shape)
        print(np.unique(predictions))
        print("===")
    elif mode == 'max':
        predictions = np.max(predictions_test, axis=0)
        print(predictions.shape)

    return predictions


def ensemble(args):
    mlist = args.modellist
    img_sizes = args.img_sizes
    val_split = args.val_split
    mode = args.mode
    id = args.id
    #thres_opt = args.thres_opt
    csv_files = args.csv_files

    if mlist:
        predictions_train, all_labels, predictions_test, test_files = create_predictions(mlist, val_split)
    else:
        reader = HDF_line_reader('input/test.h5', load_rgb=False)
        _, test_files = d.get_all_test(reader)
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
                    predictions_csv[i, j, :] = data.to_one_hot(row)
        predictions_tmp[:predictions_test.shape[0], :, :] = predictions_test
        predictions_tmp[predictions_test.shape[0]:, :, :] = predictions_csv
        predictions_test = predictions_tmp

    predictions = predict_with_ensemble(predictions_test, mode)
    result = pd.DataFrame(predictions, columns=data.labels)

    ## Check for reasonable distribution
    # Get targets from training data hdf5 file
    reader = d.HDF_line_reader('input/train.h5', load_rgb=False)
    _, targets, file_ids = d.get_all_train(reader=reader)
    df = pd.DataFrame(np.array(targets), columns=data.labels)

    # Create 2dim co occurence matrix by matrix multiplication
    df_asint = df.astype(int)
    # Count individual probabilities
    # P(A)
    summat = df_asint.values.sum(axis=0)
    summatp = summat / np.sum(summat)

    thresholds = np.arange(0.01,0.99,0.01)
    diff = np.zeros((len(thresholds)))
    for i,t in enumerate(thresholds):
        r = result.values.copy()
        r[r <= t] = 0
        r[r > t] = 1
        r = np.sum(r,axis=0)
        r = r / np.sum(r)
        difference = summatp-r
        diff[i] = np.sum(np.abs(difference))

    print(diff)
    index_min = np.argmin(diff)
    print("Index",index_min)
    thres_opt = thresholds[index_min]
    print('Threshold',thres_opt)

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
    img_sizes = []
    #mlist = ['simple_net_0.68','simple_net_0.48']
    # List with same size as mlist
    #img_sizes = [64,64]
    #csv_files = []
    csv_files = ['input/submissions/submission_1.csv', 'input/submissions/submission_blend.csv','input/submissions/subm_10fold_128.csv','input/submissions/submission_tiff.csv','input/submissions/submission_xgb.csv','input/submissions/submission_keras-2.csv']
    val_split = 0.2
    mode = 'mean'
    id = 'more_subs_with_prob_check'
    thres_opt = 0.6
    args = Namespace(val_split=val_split, modellist=mlist, img_sizes = img_sizes, mode=mode, id=id, thres_opt=thres_opt, csv_files=csv_files)

    ensemble(args)
