from multiprocessing.dummy import Pool as ThreadPool

import cv2
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from spectral import *
from tqdm import tqdm
import pdb
import data as dat

LABELS = dat.LABELS
REV_LABELS = dat.REV_LABELS

class Validation_splitter:
    '''
        Training/Validation data split utility class. Holds array with indices
        of training and validation data as defined by percentage (percentage of
        validation data) and csv to train data
    '''

    def __init__(self, path, percentage):
        '''
        @params:
        -path: Path to your .h5 file
        -percentage: Percentage of your data to be used for validation (~0.2)
        '''
        
        file = h5py.File(path, "r")
        ids = file["filenames"]
        self.row_nums = np.arange(len(ids))
        self.percentage = percentage
        self.num_fold = 0
        self.num_folds = int(1.0 / percentage)
        self.fold_size = int(len(self.row_nums) * percentage)

    def next_fold(self):
        '''
        Calls the next crossval fold.
        
        @return:
        returns True when the next fold was initiated succesfully, False otherwise.
        '''
        if self.num_folds > self.num_fold:
            if self.num_folds > self.num_fold + 1:
                select = np.arange(self.num_fold * self.fold_size, self.num_fold * self.fold_size + self.fold_size)
            else:
                select = np.arange(self.num_fold * self.fold_size, len(self.row_nums))
            self.val_idx = self.row_nums[select]

            train_select = np.full(len(self.row_nums), True)
            train_select[select] = False
            self.train_idx = self.row_nums[train_select]

            self.num_fold += 1
            return True
        else:
            return False


class HDF_line_reader:
    '''
    Reads lines from a HDF5 matrix
    '''
    def __init__(self, path, load_rgb=False, img_size=256):
        '''
        @params:
        -path: Path to your .h5 file
        -load_rgb: True if the image format is rgb, false for bgr.
        -img_size: Desired output height/width
        
        '''
        file = h5py.File(path, "r")
        self.images = file['imgs']
        self.test = False
        if 'test' in path:
            self.test = True
        else:
            self.labels = file['labels']
        self.filenames = file['filenames']
        fun = np.vectorize(lambda x: x.decode('utf-8'))
        self.filenames = fun(self.filenames)

        self.rgb = load_rgb
        self.img_size = img_size

    def read_line_hdf(self, line_num):
        '''
        Reading operator
        @params:
        linu_num: int or list of indices of which files to read
        
        @returns:
        -imgs: [len(line_num),size,size,4] array of pixel values
        -labels: [len(line_num)] np array of class labels
        -filenames: [lin(line_num)] array of file names.
        
        '''
        imgs = map(lambda x: get_rgb(x,[2,1,0]), self.images[line_num]) if self.rgb else self.images[line_num]
        if self.img_size < 256:
            pool = ThreadPool(4)
            imgs = pool.map(lambda x: cv2.resize(x, (self.img_size, self.img_size)), imgs)
            pool.terminate()

        if not self.test:
            return imgs, self.labels[line_num], self.filenames[line_num]
        else:
            return imgs, self.filenames[line_num]


def get_all_train(reader):
    '''
    Get all image files at once
    
    @params:
    -reader: HDF_line_reader(your_dataset)
    '''
    d = reader.images
    l = reader.labels
    file_ids = reader.filenames

    return d, l, file_ids


'''def get_all_val(data_dir, reader, splitter, img_size=256, load_rgb=False):
    ''
    Get entire validation set at once
    
    @params:
    -reader: CSV_reader(your_dataset)
    ''
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

    return np.array(d), np.array(l)'''


def train_generator(reader, splitter, batch_size):
    '''
    train_generator functional to be passed to keras.model.fit_generator(). Generates 
    training images with data augmentation
    
    :param reader: HDF_line_reader(your_dataset)
    :param splitter: Validation_splitter() object 
    :param batch_size: 

    :return: 
    '''

    train_idx = splitter.train_idx

    datagen = ImageDataGenerator(
        #rotation_range=15,
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

        d.extend(imgs)
        l.extend(labels)

        d = np.array(d)
        l = np.array(l)

        datagen.fit(d)

        cnt = 0
        for X_batch, Y_batch in datagen.flow(d, l, batch_size=batch_size):

            yield (X_batch, Y_batch)
            cnt += batch_size
            if cnt >= num:
                break


def val_generator(reader, splitter, batch_size):
    '''
    val_generator functional to be passed to keras.model.evaluate_generator(). Generates 
    validation images.

    :param reader: HDF_line_reader(your_dataset)
    :param splitter: Validation_splitter() object 
    :param batch_size:     
    
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
        d.extend(imgs)
        l.extend(labels)

        d = np.array(d)
        l = np.array(l)

        cnt = 0
        num_batches = len(d) / batch_size
        for batch in range(int(num_batches)):
            yield (np.array(d[cnt:cnt + batch_size]), np.array(l[cnt:cnt + batch_size]))
            cnt += batch_size
            if cnt >= num:
                break


def test_generator(reader, batch_size):
    '''
    test_generator functional to be passed to keras.model.predict_generator(). Generates 
    the prediction image data flow.
    
    :param reader: HDF_line_reader(your_dataset)
    
    '''
    
    start = 0
    l = len(reader.images)
    while True:
        # idx = val_idx[start:(start+batch_size)%len(val_idx)]
        # start += batch_size
        if start + batch_size > l:
            end = start + batch_size - l
            select = np.concatenate([np.arange(start, l).astype(int), np.arange(end).astype(int)])
        else:
            select = np.arange(start, start + batch_size).astype(int)
        
        select = list(select)
        select.sort()
        
        imgs,_ = reader.read_line_hdf(select)
        start += batch_size

        yield (np.array(imgs))
