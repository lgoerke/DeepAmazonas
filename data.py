import random
import csv
import sys
import os
import time
import numpy as np
#from PIL import Image
#import georasters as gr
from keras.preprocessing.image import ImageDataGenerator
from spectral import *
from skimage import io as skio
from sklearn.preprocessing import MinMaxScaler
import cv2
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

LABELS = {'blow_down':0,
          'bare_ground':1,
          'conventional_mine':2,
          'blooming':3,
          'cultivation':4,
          'artisinal_mine':5,
          'haze':6,
          'primary':7,
          'slash_burn':8,
          'habitation':9,
          'clear':10,
          'road':11,
          'selective_logging':11,
          'partly_cloudy':12,
          'agriculture':13,
          'water':14,
          'cloudy':15}

class Validation_splitter:
    '''
        Training/Validation data split utility class. Holds array with indices
        of training and validation data as defined by percentage (percentage of
        validation data) and csv to train data
    '''
    def __init__(self,csv_path,percentage):
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile,delimiter = ",")
            data = list(reader)
            ## Don't read header (-1)
            self.row_nums = np.arange(len(data)-1)
            np.random.shuffle(self.row_nums)
            self.percentage = percentage
            self.num_fold = 0
            self.num_folds = int(1.0/percentage)
            self.fold_size = int(len(self.row_nums)*percentage)
    
    def next_fold(self):
        if self.num_folds > self.num_fold:
            if self.num_folds > self.num_fold + 1:
                select = np.arange(self.num_fold*self.fold_size,self.num_fold*self.fold_size + self.fold_size)           
            else:
                select = np.arange(self.num_fold*self.fold_size,len(self.row_nums))
            self.val_idx = self.row_nums[select]
            self.train_idx = self.row_nums[~select]
            self.num_fold += 1
            return True
        else:
            return False

class CSV_line_reader:
    def __init__(self,csv_path):
        with open(csv_path, 'r') as csvfile:
            self.content = list(csv.reader(csvfile))

    def read_line_csv(self,line_num):
        ## Because header was deleted in indices, read next line always
        return self.content[line_num+1][0], self.content[line_num+1][1]

def load_single_tif(dir,file_path,img_size,to_255=False):
    '''
    Returns tif image with (img_size,img_size,4) shape and VI Score image with shape (img_size,img_size)
    '''
    open_path = os.path.join(dir, file_path + '.tif')
    imarray = gr.from_file(open_path)
    im = np.reshape(imarray.raster,(4,256,256))
    im = np.transpose(im,(1,2,0))
    rescaleIMG = np.reshape(im, (-1, 1))
    if to_255:
        scaler = MinMaxScaler(feature_range=(0, 255))
        rescaleIMG = scaler.fit_transform(rescaleIMG)
        img_scaled = (np.reshape(rescaleIMG, im.shape)).astype(np.uint8)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaleIMG = scaler.fit_transform(rescaleIMG)
        img_scaled = (np.reshape(rescaleIMG, im.shape)).astype(np.float32)

    # spectral module ndvi function
    vi = ndvi(im, 2, 3)

    return cv2.resize(img_scaled, (img_size, img_size)), vi

def load_tif_as_rgb(dir,file_path,img_size,to_255=False):
    '''
    Returns rgb image with (img_size,img_size,3) shape
    '''
    open_path = os.path.join(dir, file_path + '.tif')
    img = skio.imread(open_path)
    img_rgb = get_rgb(img, [2, 1, 0]) # RGB
    rescaleIMG = np.reshape(img_rgb, (-1, 1))
    if to_255:
        scaler = MinMaxScaler(feature_range=(0, 255))
        rescaleIMG = scaler.fit_transform(rescaleIMG) 
        img_scaled = (np.reshape(rescaleIMG, img_rgb.shape)).astype(np.uint8)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaleIMG = scaler.fit_transform(rescaleIMG)
        img_scaled = (np.reshape(rescaleIMG, img_rgb.shape)).astype(np.float32)
    
    return cv2.resize(img_scaled, (img_size, img_size))

def get_all_val(data_dir, reader, splitter, img_size=256, load_rgb=False):
    val_idx = splitter.val_idx
    d = []
    l = []

    for i in tqdm(val_idx, desc='Loading validation set'):
        if load_rgb:
            d.append(load_tif_as_rgb(data_dir,reader.read_line_csv(i)[0],img_size))
        else:
            loaded, _ = load_single_tif(data_dir,reader.read_line_csv(i)[0],img_size)
            d.append(loaded)
        l.append(to_one_hot(reader.read_line_csv(i)[1]))

    return d, l

def get_all_test(data_dir, img_size=256, load_rgb=False):
    files = [os.path.splitext(f)[0] for f in listdir(data_dir) if isfile(join(data_dir, f))]
    d = []

    for f in tqdm(files, desc='Loading test set'):
        if load_rgb:
            d.append(load_tif_as_rgb(data_dir,f,img_size))
        else:
            loaded, _ = load_single_tif(data_dir,f,img_size)
            d.append(loaded)

    return d

def to_one_hot(targets):

    one_hot = np.zeros(len(LABELS))
    for label in targets.split(' '):
        one_hot[LABELS[label]] = 1

    return one_hot

def train_generator(data_dir, reader, splitter, batch_size, img_size=256, load_rgb=False):
    train_idx = splitter.train_idx

    datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    num = 500 
    while True:
        sampled_idx = np.random.choice(train_idx,size=num)
        d = []
        l = []

        for i in sampled_idx:
            if load_rgb:
                d.append(load_tif_as_rgb(data_dir,reader.read_line_csv(i)[0],img_size))
            else:
                loaded, _ = load_single_tif(data_dir,reader.read_line_csv(i)[0],img_size)
                d.append(loaded)
            l.append(to_one_hot(reader.read_line_csv(i)[1]))
        d = np.array(d)
        l = np.array(l)

        datagen.fit(d)

        cnt = 0
        for X_batch, Y_batch in datagen.flow(d,l, batch_size=batch_size):
            yield (X_batch, Y_batch)
            cnt+=batch_size   
            if cnt >= num:
                break

def val_generator(data_dir, reader, splitter, batch_size, img_size=256, load_rgb=False):
    val_idx = splitter.val_idx

    start = 0
    num = 200
    times = int(num/batch_size)
    num = times * batch_size
    while True:
        #idx = val_idx[start:(start+batch_size)%len(val_idx)]
        #start += batch_size 
        if start+num > len(val_idx):
            end = start+num - len(val_idx)
            select = np.concatenate([np.arange(start,len(val_idx)).astype(int),np.arange(end).astype(int)])
        else: 
            select = np.arange(start,start+num).astype(int)
        idx = val_idx[select]
        start += num
        if start > len(val_idx): start = start - len(val_idx)

        d = []
        l = []

        for i in idx:
            if load_rgb:
                d.append(load_tif_as_rgb(data_dir,reader.read_line_csv(i)[0],img_size))
            else:
                loaded, _ = load_single_tif(data_dir,reader.read_line_csv(i)[0],img_size)
                d.append(loaded)
            l.append(to_one_hot(reader.read_line_csv(i)[1]))

        cnt = 0
        num_batches = len(d) / batch_size
        for batch in range(int(num_batches)):
            yield (np.array(d[cnt:cnt+batch_size]), np.array(l[cnt:cnt+batch_size]))
            cnt+=batch_size   
            if cnt >= num:
                break
