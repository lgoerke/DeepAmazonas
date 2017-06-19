import os
from argparse import Namespace

import keras
import utils
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.models import Sequential

from sklearn.metrics import fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import data_hdf5 as d
import data as data
from data_hdf5 import HDF_line_reader
from data_hdf5 import Validation_splitter
import tensorflow as tf


def train_logistic(predictions_train, all_labels, id):
    predictions_train = np.transpose(predictions_train, (1, 0, 2))
    predictions_train = np.reshape(predictions_train,
                                   (
                                   predictions_train.shape[0], predictions_train.shape[1] * predictions_train.shape[2]))

    logistic = OneVsRestClassifier(LogisticRegression())

    print('Predictions train', predictions_train.shape)
    print('Labels train', all_labels.shape)

    cutoff = int(len(predictions_train)*0.8)

    logistic.fit(predictions_train[:cutoff,:], np.array(all_labels)[:cutoff,:])
    s = logistic.score(predictions_train[cutoff:, :], np.array(all_labels)[cutoff:, :])
    print('Score',s)
    print('validating')
    p_valid = logistic.predict_proba(predictions_train[cutoff:,:])

    loss = fbeta_score(np.array(all_labels)[cutoff:,:], np.array(p_valid) > 0.23, beta=2, average='samples')
    print('validation loss: {}'.format(loss))
    probas = logistic.predict_proba(predictions_train[cutoff:,:])
    thres_opt = utils.optimise_f2_thresholds(np.array(all_labels)[cutoff:,:], probas)
    print(thres_opt)

    # save the model to disk
    filename = 'logistic_regr.sav'
    pickle.dump(logistic, open(filename, 'wb'))

def train_ensemble_full(predictions_train, all_labels, id):
    predictions_train = np.transpose(predictions_train, (1, 0, 2))
    predictions_train = np.reshape(predictions_train,
                                   (predictions_train.shape[0], predictions_train.shape[1] * predictions_train.shape[2]))

    model = Sequential()
    model.add(Dense(128, input_dim=(predictions_train.shape[1]), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(17, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    print('Predictions train',predictions_train.shape)
    print('Labels train', all_labels.shape)

    # define the checkpoint
    filepath = 'ensemble/weights_{epoch:02d}_{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, verbose=1)
    callbacks_list = [checkpoint]
    model.fit(predictions_train,  np.array(all_labels), epochs=50, batch_size=32, callbacks=callbacks_list)

def train_ensemble(predictions_train, all_labels, id):
    predictions_train = np.transpose(predictions_train, (1, 0, 2))
    predictions_train = np.reshape(predictions_train,
                                   (predictions_train.shape[0], predictions_train.shape[1] * predictions_train.shape[2]))

    model = Sequential()
    model.add(Dense(128, input_dim=(predictions_train.shape[1]), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(17, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    print('Predictions train',predictions_train.shape)
    print('Labels train', all_labels.shape)

    # define the checkpoint
    filepath = 'ensemble/weights_{epoch:02d}_{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, verbose=1)
    callbacks_list = [checkpoint]

    cutoff = int(len(predictions_train)*0.8)
    # Fit the model
    model.fit(predictions_train[:cutoff,:],  np.array(all_labels)[:cutoff,:], epochs=50, batch_size=32, callbacks=callbacks_list)

    p_valid = model.predict(predictions_train[cutoff:,:])

    loss = fbeta_score(np.array(all_labels)[cutoff:,:], np.array(p_valid) > 0.2, beta=2, average='samples')
    print('validation loss: {}'.format(loss))
    probas = model.predict(predictions_train[cutoff:,:])
    thres_opt = utils.optimise_f2_thresholds(np.array(all_labels)[cutoff:,:], probas)
    print(thres_opt)

def predict_with_ensemble(predictions_test,id,epoch,loss, mode='mean'):
    if mode == 'network':
        predictions_test = np.transpose(predictions_test, (1, 0, 2))
        predictions_test = np.reshape(predictions_test,
                                      (predictions_test.shape[0], predictions_test.shape[1] * predictions_test.shape[2]))

        model = Sequential()
        model.add(Dense(128, input_dim=(predictions_test.shape[1]), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(17, activation='sigmoid'))

        # define the checkpoint
        filepath = 'ensemble/weights_{}_{}_{}.hdf5'.format(id,epoch,loss)
        model.load_weights(filepath)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())

        predictions = model.predict(predictions_test)
        print(predictions.shape)
    elif mode == "regr":
        predictions_test = np.transpose(predictions_test, (1, 0, 2))
        predictions_test = np.reshape(predictions_test,
                                      (
                                      predictions_test.shape[0], predictions_test.shape[1] * predictions_test.shape[2]))
        loaded_model = pickle.load(open('logistic_regr.sav', 'rb'))
        predictions = loaded_model.predict_proba(predictions_test)
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
    model_csv_train = args.model_csv_train
    model_csv_test = args.model_csv_test
    mode = args.mode
    id = args.id
    thres_list = args.thres_list
    csv_files = args.csv_files
    chosen_weights_e = args.chosen_weights_e
    chosen_weights_l = args.chosen_weights_l
    first_run = args.first_run

    if mode == "network" or mode == "regr":

        reader_train = HDF_line_reader('input/train.h5', load_rgb=False, img_size=42)
        all_labels = reader_train.labels
        train_filenames = reader_train.filenames

        pdf = pd.DataFrame(train_filenames, columns=['image_name'])
        ldf = pd.DataFrame(np.array(all_labels), columns=data.labels)
        labeldf = pd.concat([pdf, ldf], axis=1)
        labeldf = labeldf.sort_values('image_name')
        del labeldf['image_name']

        condp2 = pickle.load(open('input/condp2.pkl', 'rb'))
        condp3 = pickle.load(open('input/condp3.pkl', 'rb'))

        cp2flat = condp2.flatten()
        cp3flat = condp3.flatten()

        cp2flat_train = np.repeat(np.reshape(cp2flat,(1,len(cp2flat))),40479, axis=0)
        cp2flat_test = np.repeat(np.reshape(cp2flat,(1,len(cp2flat))),61191, axis=0)

        cp3flat_train = np.repeat(np.reshape(cp3flat,(1,len(cp3flat))),40479, axis=0)
        cp3flat_test = np.repeat(np.reshape(cp3flat,(1,len(cp3flat))),61191, axis=0)

        if model_csv_train:

            if use_condp:
                predictions_train = np.zeros((len(model_csv_train), 40479, 5219))
                predictions_test = np.zeros((len(model_csv_test), 61191, 5219))
            else:
                predictions_train = np.zeros((len(model_csv_train), 40479, 17))
                predictions_test = np.zeros((len(model_csv_test), 61191, 17))

            tmpdf = pd.read_csv(model_csv_test[0])
            test_files = tmpdf['image_name']

            for idx, mo in enumerate(model_csv_test):
                print('Reading',idx,'model')
                df = pd.read_csv(model_csv_train[idx])
                df = df.sort_values('image_name')
                del df['image_name']
                if use_condp:
                    predictions_train[idx, :, :] = np.concatenate([np.array(df), cp2flat_train, cp3flat_train],axis=1)
                else:
                    predictions_train[idx, : ,:] = df
                df.drop(df.index, inplace=True)
                df = pd.read_csv(model_csv_test[idx])
                del df['image_name']
                if use_condp:
                    predictions_test[idx, :, :] = np.concatenate([np.array(df), cp2flat_test, cp3flat_test],axis=1)
                else:
                    predictions_test[idx, : ,:] = df
                df.drop(df.index, inplace=True)
        else:
            reader = HDF_line_reader('input/test.h5', load_rgb=False)
            _, test_files = d.get_all_test(reader)
            predictions_test = np.zeros((0, 61191, 17))

        if first_run:

            if mode == "network":
                #assert isinstance(model_csv_train, object, "Network mode needs models")
                train_ensemble(predictions_train, labeldf, id)
            elif mode == "regr":
                train_logistic(predictions_train,labeldf,id)


        if csv_files:
            if mode == "network":
                print('CSV files are ignored because of network mode')
            else:
                predictions_tmp = np.zeros(
                    (predictions_test.shape[0] + len(csv_files), predictions_test.shape[1], predictions_test.shape[2]))
                predictions_csv = np.zeros((len(csv_files), predictions_test.shape[1], predictions_test.shape[2]))
                for i, c in enumerate(csv_files):
                    df = pd.read_csv(c)
                    df.sort_values('image_name')
                    for j, row in enumerate(df['tags']):
                        predictions_csv[i, j, :] = data.to_one_hot(row)
            predictions_tmp[:predictions_test.shape[0], :, :] = predictions_test
            predictions_tmp[predictions_test.shape[0]:, :, :] = predictions_csv
            predictions_test = predictions_tmp

        if not first_run:

            predictions = predict_with_ensemble(predictions_test,id,chosen_weights_e,chosen_weights_l, mode)
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

            # thresholds = np.arange(0.01, 0.99, 0.01)
            # diff = np.zeros((len(thresholds)))
            # for i, t in enumerate(thresholds):
            #     r = result.values.copy()
            #     r[r <= t] = 0
            #     r[r > t] = 1
            #     r = np.sum(r, axis=0)
            #     r = r / np.sum(r)
            #     difference = summatp - r
            #     diff[i] = np.sum(np.abs(difference))
            #
            # print(diff)
            # index_min = np.argmin(diff)
            # print("Index", index_min)
            # thres_opt = thresholds[index_min]
            # print('Threshold', thres_opt)

            preds = []
            print('Create csv')
            for i in tqdm(range(result.shape[0]), miniters=1000):
                a = result.ix[[i]]
                a = a.apply(lambda x: x > thres_list, axis=1)
                a = a.transpose()
                a = a.loc[a[i] == True]
                ' '.join(list(a.index))
                preds.append(' '.join(list(a.index)))
            print('Done')

            df = pd.DataFrame(np.zeros((61191, 2)), columns=['image_name', 'tags'])
            df['image_name'] = test_files
            df['tags'] = preds

            df.to_csv('ensemble/submission_ensemble_{}.csv'.format(id), index=False)

    else:

        reader_train = HDF_line_reader('input/train.h5', load_rgb=False, img_size=42)
        all_labels = reader_train.labels
        train_filenames = reader_train.filenames

        pdf = pd.DataFrame(train_filenames, columns=['image_name'])
        ldf = pd.DataFrame(np.array(all_labels), columns=data.labels)
        labeldf = pd.concat([pdf, ldf], axis=1)

        if model_csv_train:

            predictions_train = np.zeros((len(model_csv_test), 40479, 17))
            predictions_test = np.zeros((len(model_csv_train), 61191, 17))

            tmpdf = pd.read_csv(model_csv_test[0])
            test_files = tmpdf['image_name']

            for idx, mo in enumerate(model_csv_test):
                df = pd.read_csv(model_csv_train[idx])
                df = df.sort_values('image_name')
                del df['image_name']
                predictions_train[idx, :, :] = df
                df.drop(df.index, inplace=True)
                df = pd.read_csv(model_csv_test[idx])
                del df['image_name']
                predictions_test[idx, :, :] = df
                df.drop(df.index, inplace=True)
        else:
            reader = HDF_line_reader('input/test.h5', load_rgb=False)
            _, test_files = d.get_all_test(reader)
            predictions_test = np.zeros((0, 61191, 17))

        if mode == "network":
            #assert isinstance(model_csv_train, object, "Network mode needs models")
            train_ensemble(predictions_train, labeldf, id)

        if csv_files:
            if mode == "network":
                print('CSV files are ignored because of network mode')
            else:
                predictions_tmp = np.zeros(
                    (predictions_test.shape[0] + len(csv_files), predictions_test.shape[1], predictions_test.shape[2]))
                predictions_csv = np.zeros((len(csv_files), predictions_test.shape[1], predictions_test.shape[2]))
                for i, c in enumerate(csv_files):
                    df = pd.read_csv(c)
                    df.sort_values('image_name')
                    for j, row in enumerate(df['tags']):
                        predictions_csv[i, j, :] = data.to_one_hot(row)
            predictions_tmp[:predictions_test.shape[0], :, :] = predictions_test
            predictions_tmp[predictions_test.shape[0]:, :, :] = predictions_csv
            predictions_test = predictions_tmp

        predictions = predict_with_ensemble(predictions_test,id, mode)
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

        thresholds = np.arange(0.01, 0.99, 0.01)
        diff = np.zeros((len(thresholds)))
        for i, t in enumerate(thresholds):
            r = result.values.copy()
            r[r <= t] = 0
            r[r > t] = 1
            r = np.sum(r, axis=0)
            r = r / np.sum(r)
            difference = summatp - r
            diff[i] = np.sum(np.abs(difference))

        print(diff)
        index_min = np.argmin(diff)
        print("Index", index_min)
        thres_opt = thresholds[index_min]
        print('Threshold', thres_opt)

        preds = []
        print('Create csv')
        for i in tqdm(range(result.shape[0]), miniters=1000):
            a = result.ix[[i]]
            a = a.apply(lambda x: x > thres_opt, axis=1)
            a = a.transpose()
            a = a.loc[a[i] == True]
            ' '.join(list(a.index))
            preds.append(' '.join(list(a.index)))
        print('Done')

        df = pd.DataFrame(np.zeros((61191, 2)), columns=['image_name', 'tags'])
        df['image_name'] = test_files
        df['tags'] = preds

        df.to_csv('ensemble/submission_ensemble_{}.csv'.format(id), index=False)


if __name__ == '__main__':
    model_csv_train = ['ensemble/train_preds_dense.csv', 'ensemble/xgb_pred_probs_train.csv']
    model_csv_test = ['ensemble/test_preds_dense.csv', 'ensemble/xgb_pred_probs_test.csv']

    #'ensemble/vgg_pred_train.csv'
    #'ensemble/vgg_pred_test.csv'
    # List with same size as mlist
    csv_files = []
    # csv_files = ['input/submissions/submission_1.csv', 'input/submissions/submission_blend.csv']
    # mlist = []
    # img_sizes = []
    # csv_files = ['input/submissions/submission_1.csv', 'input/submissions/submission_blend.csv','input/submissions/subm_10fold_128.csv','input/submissions/submission_tiff.csv','input/submissions/submission_xgb.csv','input/submissions/submission_keras-2.csv']
    mode = 'network'
    id = 'xgb_dense_nn3_before'
    chosen_weights_e = '46'
    chosen_weights_l ='0.08'
    first_run = False
    use_condp = True
    thres_list = [0.25, 0.35, 0.42, 0.2, 0.2, 0.41, 0.19, 0.2, 0.45, 0.09, 0.14, 0.12, 0.43, 0.25, 0.18, 0.28, 0.05] 
    args = Namespace(model_csv_train=model_csv_train, model_csv_test=model_csv_test, mode=mode, id=id,
                     thres_list=thres_list, csv_files=csv_files,chosen_weights_e=chosen_weights_e,chosen_weights_l=chosen_weights_l,first_run=first_run)

    ensemble(args)
