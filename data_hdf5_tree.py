from multiprocessing.dummy import Pool as ThreadPool

import cv2
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from spectral import *
from tqdm import tqdm
import utils
import copy
import data_hdf5
from data_hdf5 import Validation_splitter, HDF_line_reader, test_generator
import data as dat

LABELS = dat.LABELS



def train_generator(reader, splitter, batch_size, use_labels = [], included_columns=[], new_columns={}):
    '''
    train_generator functional to be passed to keras.model.fit_generator(). Generates 
    training images with data augmentation with additional functionality for a 
    sequential classifier set-up. 
    
    :param reader: HDF_line_reader(your_dataset)
    :param splitter: Validation_splitter() object 
    :param batch_size: 
    :param use_labels: Array of y-values to use in this training session ['forest','clouds']
    :param included_columns: List of types to include in this training session ['clear']
    :param new_column: add a new column to integrate information ({'weather':['clear','cloudy']})

    '''

    train_idx = splitter.train_idx

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    start = 0
    num = 1000
    times = int(num / batch_size)
    num = times * batch_size
    while True:
        if start + num > len(train_idx):
            end = start + num - len(train_idx)
            select = np.concatenate([np.arange(start, len(train_idx)).astype(int), np.arange(end).astype(int)])
        else:
            select = np.arange(start, start + num).astype(int)
        idx = train_idx[select]

        idx.sort()
        start += num
        if start > len(train_idx): start = start - len(train_idx)

        
        d = []
        l = []
        
        imgs, labels, _ = reader.read_line_hdf(list(idx))
        LABELS = copy.copy(dat.LABELS)
        if new_columns:
            for key, value in new_columns.items():
                indeces = []
                for c in value:
                    indeces.append(LABELS[c])
                # TODO think about if we need the column name
                LABELS[key] = len(LABELS.keys())
                labels = np.append(labels, np.expand_dims(np.any(labels[:, indeces], axis=1), axis=1), axis=1)
                

                
        if included_columns:
            #create a  mask to cut out all unwanted images
            mask_idx = np.zeros(len(labels))
            
            for cls in (included_columns):
                #print(mask_idx.shape,labels.shape,LABELS,cls)
                mask_idx = np.logical_or(labels[:,LABELS[cls]],mask_idx)


            
        if use_labels :
            # Cut away all uninportant class labels from y:
            lbls = np.zeros((len(LABELS.keys())))
            for lbl in (use_labels):
                lbls[LABELS[lbl]] = True  # TODO: make 1-dimensional
            labels = labels[:, lbls.astype(np.bool)]
        
        
            
        d.extend(imgs)
        l.extend(labels)


        
        d = np.array(d)
        l = np.array(l)
        
        
        
        if included_columns:
            d = d[mask_idx]
            l = l[mask_idx]

        datagen.fit(d)
                
        if new_columns:        
            for key,_ in new_columns.items():
                del LABELS[key]
                
        #print(d.shape, l.shape, mask_idx.sum())
        
        cnt = 0
        advance = 0
        for X_batch, Y_batch in datagen.flow(d, l, batch_size=batch_size):

            #print('training batch...', X_batch.shape)   
            yield (X_batch, Y_batch)
            cnt += Y_batch.shape[0]
            advance += batch_size
            
            if advance >= num :
                
                break



def val_generator(reader, splitter, batch_size, use_labels = [],included_columns=[], new_columns={}):
    '''
    :param reader: HDF_line_reader(your_dataset)
    :param splitter: Validation_splitter() object 
    :param batch_size: 
    :param use_labels: Array of y-values to use in this training session ['forest','clouds']
    :param included_columns: List of types to include in this training session ['clear']
    :param new_column: add a new column to integrate information ({'weather':['clear','cloudy']})
    
    '''
    val_idx = splitter.val_idx


    start = 0
    num = 1000
    times = int(num / batch_size)
    num = times * batch_size
    while True:
        # idx = val_idx[start:(start+batch_size)%len(val_idx)]
        # start += batch_size
        if start + num > len(val_idx):
            end = start + num - len(val_idx)
            select = np.concatenate([np.arange(start, len(val_idx)).astype(int), np.arange(end).astype(int)])
        else:
            select = np.arange(start, start + num).astype(int)
        idx = val_idx[select]
        idx.sort()
        start += num
        if start > len(val_idx): start = start - len(val_idx)

        d = []
        l = []

        imgs, labels, _ = reader.read_line_hdf(list(idx))
        
        LABELS = copy.copy(dat.LABELS)
        if new_columns:
            for key, value in new_columns.items():
                indeces = []
                for c in value:
                    indeces.append(LABELS[c])
                # TODO think about if we need the column name
                LABELS[key] = len(LABELS.keys())
                labels = np.append(labels, np.expand_dims(np.any(labels[:, indeces], axis=1), axis=1), axis=1)
                
        if included_columns:
            #create a  mask to cut out all unwanted images
            mask_idx = np.zeros(len(idx))
            
            for cls in (included_columns):
                #print(mask_idx.shape,labels.shape,LABELS,cls)
                try:
                    mask_idx = np.logical_or(labels[:,LABELS[cls]],mask_idx) 
                    #print('good!')
                except: break
            
            
        if use_labels :
            # Cut away all uninportant class labels from y:
            lbls = np.zeros((len(LABELS.keys())))
            for lbl in (use_labels):
                lbls[LABELS[lbl]] = True  # TODO: make 1-dimensional
            labels = labels[:, lbls.astype(np.bool)]

        d.extend(imgs)
        l.extend(labels)

        d = np.array(d)
        l = np.array(l)
        
        
        if included_columns:
            d = d[mask_idx]
            l = l[mask_idx]
        
        if new_columns:        
            for key,_ in new_columns.items():
                del LABELS[key]


        
        cnt = 0
        advance = 0
        num_batches = len(d) / batch_size
        if num_batches == 0:
            print('no batches...')
            break
        for batch in range(int(num_batches)):
            #print('validation batch...', cnt, num, batch_size) 
            yield (np.array(d[cnt:cnt + batch_size]), np.array(l[cnt:cnt + batch_size]))
            cnt += batch_size
            if cnt >= num:
                break
