
# coding: utf-8

# Keras + CV
# 
# Thanks @anokas for the starter code at https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/simple-keras-starter/

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import cv2
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import gdal
from tqdm import tqdm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time


# Pre-processing the train and test data

x_train = []
x_test = []
y_train = []

#n_train_images = 40479
df_train = pd.read_csv('../data/train_v2.csv')
#df_train = pd.read_csv('../data/train_v2.csv', nrows=n_train_images)
#df_train = pd.read_csv('../data/train_100.csv')
#n_test_images = 2000
df_test = pd.read_csv('../data/sample_submission_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

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

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

resize_to = (64,64)

#for f, tags in tqdm(df_train.values[:18000], miniters=1000):
for f, tags in tqdm(df_train.values, miniters=1000):
    img = io.imread('../data/train-tif-v2/{}.tif'.format(f))
    #img = io.imread('../data/train-tif_100/{}.tif'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, resize_to))
    y_train.append(targets)

check_dependencies=False
if check_dependencies:
    cloudy_partiallycloudy_clear_haze=[]
    cloudy_primary=[]
    for i in range(len(y_train)):
        cloudy_partiallycloudy_clear_haze.append(y_train[i][16]+y_train[i][13]+y_train[i][10]+y_train[i][6])
        cloudy_primary.append(y_train[i][1]+y_train[i][7]+y_train[i][16])
    cloudy_partiallycloudy_clear_haze=np.array(cloudy_partiallycloudy_clear_haze)
    cloudy_primary=np.array(cloudy_primary)
    np.min(cloudy_partiallycloudy_clear_haze)
    np.max(cloudy_partiallycloudy_clear_haze)
    np.min(cloudy_primary)
    np.max(cloudy_primary)

#xjpg_train=[]
#yjpg_train=[]
#for f, tags in tqdm(df_train.values, miniters=1000):
#    img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
#    #img = cv2.imread('../data/train-jpg_100/{}.jpg'.format(f))
#    targets = np.zeros(17)
#    for t in tags.split(' '):
#        targets[label_map[t]] = 1
#    xjpg_train.append(cv2.resize(img, resize_to))
#    yjpg_train.append(targets)

for f, tags in tqdm(df_test.values, miniters=1000):
    img = io.imread('../data/test-tif-v2/{}.tif'.format(f))
    #img = io.imread('../data/test-tif_100/{}.tif'.format(f))
    #img = cv2.imread('../data/test-jpg/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, resize_to))

show_images=False
if show_images:
    plt.figure(figsize=(12,8))
    plt.subplot(141)
    plt.imshow(x_train[1][ :, :, 0])
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(x_train[1][ :, :, 1])
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(x_train[1][ :, :, 2])
    plt.colorbar()
    plt.subplot(144)
    plt.imshow(x_train[1][ :, :, 3])
    plt.colorbar()
    plt.figure(figsize=(12,8))
    plt.subplot(141)
    plt.imshow(x_train[4][ :, :, 0])
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(x_train[4][ :, :, 1])
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(x_train[4][ :, :, 2])
    plt.colorbar()
    plt.subplot(144)
    plt.imshow(x_train[4][ :, :, 3])
    plt.colorbar()
    plt.show()

#yjpg_train = np.array(yjpg_train[:1000], np.uint8)
#xjpg_train = np.array(xjpg_train[:1000], np.float32) / 255.

x_max = np.max(x_train)

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32) / x_max
x_test  = np.array(x_test, np.float32) / x_max

print(x_train.shape)
print(y_train.shape)

# Transpose the data if use Theano

#x_train = x_train.transpose((0, 3, 1, 2))
#x_test = x_test.transpose((0, 3, 1, 2))

# Create n-folds cross-validation

# https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
import numpy as np
from sklearn.metrics import fbeta_score

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

from keras.layers.normalization import BatchNormalization

nfolds = 3

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train =[]

kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)

for train_index, valid_index in kf:
        start_time_model_fitting = time.time()

        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[valid_index]
        Y_valid = y_train[valid_index]

        #x_test = np.array(x_test[est_index], np.float32) / x_max

        #X_train = x_train[train_index]
        #Y_train = y_train[train_index]
        #X_valid = x_train[test_index]
        #Y_valid = y_train[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        
        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
        
        model = Sequential()
        model.add(BatchNormalization(input_shape=(resize_to[0], resize_to[1], 4))) # 64
        model.add(Conv2D(8, 1, 1, activation='relu')) # 64
        model.add(Conv2D(16, 3, 3, activation='relu')) # 62
        model.add(MaxPooling2D(pool_size=(2, 2))) # 31
        model.add(Conv2D(32, 3, 3, activation='relu')) # 29
        model.add(MaxPooling2D(pool_size=(2, 2))) # 14
        model.add(Dropout(0.25))
        model.add(Conv2D(64, 3, 3, activation='relu')) # 12
        model.add(MaxPooling2D(pool_size=(2, 2))) # 6
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', 
                      optimizer='adam',
                      metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]
        
        model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=128,verbose=2, nb_epoch=10,callbacks=callbacks,
                  shuffle=True)
        
        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)
        
        p_valid = model.predict(X_valid, batch_size = 128, verbose=2)
        print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
        print("Optimizing prediction threshold")
        thres_opt = optimise_f2_thresholds(Y_valid, p_valid)
        print(thres_opt)
        
        p_test = model.predict(x_train, batch_size = 128, verbose=2)
        yfull_train.append(p_test)
        
        p_test = model.predict(x_test, batch_size = 128, verbose=2)
        yfull_test.append(p_test)


# Averaging the prediction from each fold

result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels)
#result


# Output prediction for submission

from tqdm import tqdm
#thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.23, 0.24, 0.12, 0.14, 0.25, 0.26, 0.16]
thres = thres_opt
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > thres, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test['tags'] = preds

df_test.to_csv('submission_keras.csv', index=False)


# Averaging the prediction from each fold

result = np.array(yfull_train[0])
for i in range(1, nfolds):
    result += np.array(yfull_train[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels)

# Output prediction for submission

from tqdm import tqdm
thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.23, 0.24, 0.12, 0.14, 0.25, 0.26, 0.16]
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > thres, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_train['tags'] = preds

df_train.to_csv('train_predict_keras.csv', index=False)

tt

