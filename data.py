import random
import csv
import sys
import os
import time
import numpy as np
from PIL import Image
from skimage import io
import georasters as gr
from keras.preprocessing.image import ImageDataGenerator

class Validation_splitter:
    '''
        Training/Validation data split utility class. Holds array with indices
        of training and validation data as defined by percentage and csv to train data
    '''
    def __init__(self,csv_path,percentage):
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile,delimiter = ",")
            data = list(reader)
            row_nums = np.arange(len(data))
            np.random.shuffle(row_nums)
            self.train_idx = row_nums[:int(len(row_nums)*percentage)]
            self.val_idx = row_nums[int(len(row_nums)*percentage):]

class CSV_line_reader:
    def __init__(self,csv_path):
        with open(csv_path, 'r') as csvfile:
            self.content = list(csv.reader(csvfile))

    def read_line_csv(self,line_num):
        return self.content[line_num][0], self.content[line_num][1] 

def load_single_tif(dir,file_path,img_size,to_255=False):
    open_path = os.path.join(dir, file_path + '.tif')
    #imarray = io.imread(open_path)
    
    #im = Image.open(open_path)
    #imarray = numpy.array(im)
    imarray = gr.from_file(open_path)
    im = np.reshape(imarray.raster,(4,256,256))
    im = np.transpose(im,(1,2,0))

    if to_255:
        scaler = MinMaxScaler(feature_range=(0, 255))
        rescaleIMG = scaler.fit_transform(im)
        im = im.astype(np.uint8)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaleIMG = scaler.fit_transform(im)
        im = im.astype(np.float32)

    return cv2.resize(im, (img_size, img_size))

def load_tif_as_rgb(dir,file_path,img_size,to_255=False):
    open_path = os.path.join(dir, file_path + '.tif')
    img = io.imread(open_path)
    img_rgb = get_rgb(img, [2, 1, 0]) # RGB
    # rescaling to 0-255 range - uint8 for display
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

def calibrate_image(image):
    # TODO
    return image

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
                d.append(load_single_tif(data_dir,reader.read_line_csv(i)[0],img_size))
            l.append(reader.read_line_csv(i)[1])
        d = np.array(d)
        l = np.array(l)

        datagen.fit(d)

        cnt = 0
        for X_batch, Y_batch in datagen.flow(d,l, batch_size=batch_size):
            yield (X_batch, Y_batch)
            cnt+=batch_size   
            if cnt == num:
                break
                 
                 
def val_generator(data_dir, reader, splitter, batch_size, img_size=256, load_rgb=False):
    val_idx = splitter.val_idx

    start = 0
    while True:
        idx = val_idx[start:(start+batch_size)%len(val_idx)]
        start += batch_size 
        if start > len(val_idx): start = 0

        d = []
        l = []

        for i in idx:
            if load_rgb:
                d.append(load_tif_as_rgb(data_dir,reader.read_line_csv(i)[0],img_size))
            else:
                d.append(load_single_tif(data_dir,reader.read_line_csv(i)[0],img_size))
            l.append(reader.read_line_csv(i)[1])

        yield (np.array(d),np.array(l))
    