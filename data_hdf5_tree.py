from multiprocessing.dummy import Pool as ThreadPool

import cv2
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from spectral import *
from tqdm import tqdm

import data as dat

LABELS = dat.LABELS


class Validation_splitter:
    '''
        Training/Validation data split utility class. Holds array with indices
        of training and validation data as defined by percentage (percentage of
        validation data) and csv to train data
    '''

    def __init__(self, path, percentage, mask=None):

        file = h5py.File(path, "r")
        ids = file["filenames"]
        self.mask = mask
        self.row_nums = np.arange(len(ids))
        self.percentage = percentage
        self.num_fold = 0
        self.num_folds = int(1.0 / percentage)
        self.fold_size = int(len(self.row_nums) * percentage)

    def next_fold(self):
        if self.num_folds > self.num_fold:
            if self.num_folds > self.num_fold + 1:
                select = np.arange(self.num_fold * self.fold_size, self.num_fold * self.fold_size + self.fold_size)
            else:
                select = np.arange(self.num_fold * self.fold_size, len(self.row_nums))
            self.val_idx = self.row_nums[select]

            train_select = np.full(len(self.row_nums), True)
            train_select[select] = False
            self.train_idx = self.row_nums[train_select]

            if self.mask:
                self.current_train_mask = self.mask[train_select]
                self.current_val_mask = self.mask[select]
            else:
                self.current_train_mask = None
                self.current_val_mask = None
            self.num_fold += 1
            return True
        else:
            return False


class HDF_line_reader:
    def __init__(self, path, load_rgb=False, img_size=256):
        file = h5py.File(path, "r")
        self.images = file['imgs']
        self.test = False
        if 'test' in path:
            self.test = True
        else:
            self.labels = file['labels']
        self.filenames = file['filenames']
        self.rgb = load_rgb
        self.img_size = img_size

    def read_line_hdf(self, line_num, img_size=256):
        imgs = get_rgb(self.images[line_num], [2, 1, 0]) if self.rgb else self.images[line_num]

        if self.img_size < 256:
            pool = ThreadPool(4)
            imgs = pool.map(lambda x: cv2.resize(x, (self.img_size, self.img_size)), imgs)

        if not self.test:
            return imgs, self.labels[line_num], self.filenames[line_num]
        return imgs, self.filenames[line_num]


def get_all_train(reader):
    d = reader.images
    l = reader.labels
    file_ids = reader.filenames

    return d, l, file_ids


def get_all_val(data_dir, reader, splitter, img_size=256, load_rgb=False):
    val_idx = splitter.val_idx
    d = []
    l = []

    for i in tqdm(val_idx, desc='Loading validation set'):
        if load_rgb:
            d.append(dat.load_tif_as_rgb(data_dir, reader.read_line_csv(i)[0], img_size))
        else:
            loaded, _ = dat.load_single_tif(data_dir, reader.read_line_csv(i)[0], img_size)
            d.append(loaded)
        l.append(dat.to_one_hot(reader.read_line_csv(i)[1]))

    return np.array(d), np.array(l)


def train_generator(reader, splitter, batch_size, use_labels = [], included_columns=[], new_columns={}):
    '''
    
    :param reader: 
    :param splitter: 
    :param batch_size: 
    :param included_columns: If empty, we want to include all columns
    :return: 
    '''

    train_idx = splitter.train_idx
    current_train_mask = splitter.current_train_mask

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
        if current_train_mask:
            mask_idx = current_train_mask[select]
        idx.sort()
        start += num
        if start > len(train_idx): start = start - len(train_idx)

        
        d = []
        l = []

        imgs, labels, _ = reader.read_line_hdf(list(idx))
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

        datagen.fit(d)
        
        if new_columns:        
            for key,_ in new_columns.items():
                del LABELS[key]
        
        cnt = 0
        advance = 0
        for X_batch, Y_batch in datagen.flow(d, l, batch_size=batch_size):
            if current_train_mask or included_columns:

                X_batch = X_batch[mask_idx[advance:(advance+batch_size)]]
                Y_batch = Y_batch[mask_idx[advance:(advance+batch_size)]]

            yield (X_batch, Y_batch)
            cnt += Y_batch.shape[0]
            advance += batch_size
            if cnt >= num:
                break


def val_generator(reader, splitter, batch_size, use_labels = [],included_columns=[], new_columns={}):
    val_idx = splitter.val_idx
    current_train_mask = splitter.current_train_mask


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
        
        if new_columns:        
            for key,_ in new_columns.items():
                del LABELS[key]


        
        cnt = 0
        advance = 0
        num_batches = len(d) / batch_size
        for batch in range(int(num_batches)):
            
            d_batch = np.array(d[advance:advance+batch_size])
            l_batch = np.array(l[advance:advance+batch_size]) 
                           
            if current_train_mask or included_columns:
                d_batch = d_batch[mask_idx[advance:advance+batch_size]]
                l_batch = l_batch[mask_idx[advance:advance+batch_size]]

            yield(d_batch,l_batch)
            cnt += l_batch.shape[0]
            advance+=batch_size
            if cnt >= num: 
                break


def test_generator(reader, batch_size):
    start = 0
    while True:
        # idx = val_idx[start:(start+batch_size)%len(val_idx)]
        # start += batch_size
        imgs,_,_ = reader.read_line_hdf(range(start, start + batch_size))
        start += batch_size

        yield (np.array(imgs))
